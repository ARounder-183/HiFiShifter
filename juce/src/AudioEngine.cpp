#include "backend/MixedEnvelopeIO.h"
#include "NativeSourceTimeMap.h"
#include "NativePitchIdentity.h"
#include "NativeSharedEnvelope.h"
#include "AudioEngine.h"
#include "ClipParts.h"
#include "TrackGainEnvelope.h"
#include "StartupLog.h"
#include "backend/MelodyneProvider.h"
#include "backend/DiffSingerRenderer.h"
#include "backend/HifisamplerFlags.h"
#include "backend/UtauPitchTimeline.h"
#include "UtauPointCurve.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <unordered_set>

namespace hachi
{
namespace
{
double automaticUtauPitchTransitionInset(double leftDuration, double rightDuration)
{
    return std::max(0.002, std::min({ 0.020,
        std::max(0.01, leftDuration) * 0.20,
        std::max(0.01, rightDuration) * 0.20 }));
}

std::optional<std::pair<float, float>> contourAt(const NoteData& note, double localSeconds)
{
    if (note.nativeUnpitched) return std::nullopt;
    if (note.contour.empty()) return std::pair { 0.0f, 0.0f };
    const auto right = std::lower_bound(note.contour.begin(), note.contour.end(), localSeconds,
        [](const PitchPoint& point, double time) { return point.timeSeconds < time; });
    const auto rightIndex = right == note.contour.end()
        ? note.contour.size() - 1 : static_cast<std::size_t>(std::distance(note.contour.begin(), right));
    const auto leftIndex = rightIndex > 0 && note.contour[rightIndex].timeSeconds > localSeconds
        ? rightIndex - 1 : rightIndex;
    const auto& left = note.contour[leftIndex];
    const auto& next = note.contour[rightIndex];
    if (!left.voiced || !next.voiced) return std::nullopt;
    const auto amount = next.timeSeconds > left.timeSeconds
        ? static_cast<float>(juce::jlimit(0.0, 1.0,
            (localSeconds - left.timeSeconds) / (next.timeSeconds - left.timeSeconds))) : 0.0f;
    const auto source = left.relativeCents
        + (next.relativeCents - left.relativeCents) * amount;
    const auto leftTarget = renderedPitchCents(note, left);
    const auto rightTarget = renderedPitchCents(note, next);
    return std::pair { source, leftTarget + (rightTarget - leftTarget) * amount };
}

float amplitudeGainAt(const std::vector<AmplitudeEnvelopePoint>& envelope,
                      double localSeconds, float basePercent = 100.0f)
{
    // The base value scales the whole envelope, and an empty envelope is an
    // implied flat 0 dB line that the base raises just the same.  Everything
    // is done on the linear gain so the -60 dB floor and +12 dB ceiling match
    // the drawn envelope's own clamps.
    const auto factor = juce::jlimit(0.0f, 200.0f, basePercent) / 100.0f;
    const auto apply = [factor](float linearGain)
    {
        if (factor <= 1.0e-6f) return 0.0f;
        const auto scaled = linearGain * factor;
        if (scaled <= 1.0e-4f) return 0.0f;
        return juce::jlimit(0.0f, 3.981072f, scaled); // +12 dB ceiling.
    };
    if (envelope.empty()) return apply(1.0f);
    const auto right = std::lower_bound(envelope.begin(), envelope.end(), localSeconds,
        [](const auto& point, double time) { return point.timeSeconds < time; });
    if (right == envelope.begin())
        return apply(backend::envelopeGainFromDb(right->gainDb));
    if (right == envelope.end())
        return apply(backend::envelopeGainFromDb(envelope.back().gainDb));
    const auto& left = *(right - 1);
    const auto span = right->timeSeconds - left.timeSeconds;
    const auto amount = span > 1.0e-9
        ? static_cast<float>(juce::jlimit(0.0, 1.0,
            (localSeconds - left.timeSeconds) / span)) : 0.0f;
    const auto db = backend::envelopeDbBetween(left.gainDb,right->gainDb,amount,left.linearToNext);
    return apply(backend::envelopeGainFromDb(db));
}

std::pair<float, float> panGains(float pan, bool mono)
{
    pan = juce::jlimit(-1.0f, 1.0f, pan);
    if (mono)
        return { std::sqrt(0.5f * (1.0f - pan)),
                 std::sqrt(0.5f * (1.0f + pan)) };
    // Stereo tracks use a balance law: centre must preserve both source
    // channels at unity instead of applying an unintended -3 dB attenuation.
    return { pan > 0.0f ? std::sqrt(1.0f - pan) : 1.0f,
             pan < 0.0f ? std::sqrt(1.0f + pan) : 1.0f };
}

void applyNativeSharedPitch(backend::Mld5FileRenderRequest& request, const ClipData& clip,
    const SharedPitchLines& lines, double requestOffset = 0.0)
{
    const auto period = request.framePeriodMs / 1000.0;
    for (std::size_t frame = 0; frame < request.targetMidi.size(); ++frame)
    {
        if (!(request.targetMidi[frame] > 0.0f)) continue; // retain the unvoiced mask
        const auto local = std::min(request.targetDurationSeconds, frame * period) - requestOffset;
        for (const auto& note : clip.notes)
        {
            if (local < note.startSeconds - 1.0e-9
                || local > note.startSeconds + note.durationSeconds + 1.0e-9) continue;
            const auto* member = lines.memberFor(note.id);
            if (!member || !member->nativeSharedCurve || !member->renderSharedCurve) continue;
            const auto absoluteTime = clip.startSeconds + local;
            if (member->connectionPitchOnly && std::none_of(member->connectionPitchWindows.begin(),member->connectionPitchWindows.end(),
                [&](const auto& window){return absoluteTime>=window.getStart()-1.0e-9&&absoluteTime<=window.getEnd()+1.0e-9;})) continue;
            request.targetMidi[frame] = member->line->midiAt(clip.startSeconds + local)
                + static_cast<float>(vibratoCentsAt(note,
                    juce::jlimit(0.0, note.durationSeconds, local - note.startSeconds)) / 100.0);
            break;
        }
    }
}

// joinedStart / joinedEnd say that the clip meets its neighbour there with a
// Melodyne pitch join; see AudioEngine::clipsJoinAt.
backend::Mld5FileRenderRequest makeRenderRequest(const ClipData& clip, const TrackData& track,
                                                  const juce::File& hifiganModelDirectory,
                                                  const backend::OrtExecutionConfig& inference,
                                                  bool joinedStart = false,
                                                  bool joinedEnd = false,
                                                  const SharedPitchLines* sharedLines = nullptr,
                                                  const NativeSharedEnvelopes* sharedEnvelopes = nullptr)
{
    backend::Mld5FileRenderRequest request;
    request.sourceFile = clip.sourceFile;
    request.sourceOffsetSeconds = clip.sourceOffsetSeconds;
    request.sourceDurationSeconds = clip.sourceDurationSeconds > 1.0e-9
        ? clip.sourceDurationSeconds : clip.durationSeconds;
    request.targetDurationSeconds = clip.durationSeconds;
    request.preserveUneditedSource = std::all_of(clip.notes.begin(), clip.notes.end(), [](const auto& note)
    { return nativeSourcePitchIsKnown(note) || nativePendingPitchIsNeutral(note); });
    request.hifiganModelDirectory = hifiganModelDirectory;
    request.inference = inference;
    // The neural decoder fades each edge over 3 ms so that a clip boundary
    // whose phase is unrelated to the next one does not click.  A joined seam
    // is not one of those -- the mixer crossfades it -- and baking a 3 ms fade
    // into both sides of every join dips the level at each of them.  A bare
    // de-click is enough there.
    if (track.pitchAlgorithm == PitchAlgorithm::nsfHifigan)
    {
        if (joinedStart) request.neuralGuardStartSeconds = 0.0005f;
        if (joinedEnd) request.neuralGuardEndSeconds = 0.0005f;
    }
    switch (track.pitchAlgorithm)
    {
        case PitchAlgorithm::nsfHifigan:
            request.pitchBackend = backend::PitchRenderBackend::nsfHifigan;
            break;
        case PitchAlgorithm::mld3:
            request.pitchBackend = backend::PitchRenderBackend::mld3;
            break;
        case PitchAlgorithm::world:
            request.pitchBackend = backend::PitchRenderBackend::world;
            break;
        case PitchAlgorithm::vocalShifter:
            request.pitchBackend = backend::PitchRenderBackend::vslib;
            break;
        case PitchAlgorithm::llsm2:
            request.pitchBackend = backend::PitchRenderBackend::llsm2;
            break;
        case PitchAlgorithm::utau:
        case PitchAlgorithm::mld5:
        default:
            request.pitchBackend = backend::PitchRenderBackend::mld5;
            break;
    }
    request.stretchAlgorithm = static_cast<int>(track.stretchAlgorithm);
    request.normalizeVolume = track.normalizeVolume;
    request.nsfSmoothPitchTransitions = track.nsfSmoothPitchTransitions;
    request.nsfNoiseProtection = track.nsfNoiseProtection;
    request.nsfIncrementalRender = track.nsfNoiseProtection;
    // Decoding a clip on its own can leave its level below the source's; the
    // phrase-at-a-time order does not have that problem, because the model sees
    // the whole phrase.  The floor was raised in the per-clip order to
    // compensate -- but a splice-first reference never raises it, so per-clip
    // flooring is exactly what pulled a process-then-splice render away from a
    // splice-first bounce (each of many short clips floored independently).
    // Match the reference's behaviour: leave the neural level untouched here.
    request.matchNsfSourceLevel = false;
    for (const auto& point : nativeSourceTimeMap(clip))
        request.timeMap.push_back({ point.targetSeconds, point.sourceSeconds });
    constexpr auto framePeriodSeconds = 0.005;
    request.framePeriodMs = framePeriodSeconds * 1000.0;
    const auto frameCount = std::max(2, static_cast<int>(std::ceil(clip.durationSeconds
                                                                   / framePeriodSeconds)) + 1);
    request.sourceMidi.resize(static_cast<std::size_t>(frameCount), 0.0f);
    request.targetMidi.resize(static_cast<std::size_t>(frameCount), 0.0f);
    request.formantSemitones.resize(static_cast<std::size_t>(frameCount), 0.0f);
    const auto ownEnvelopes = sharedEnvelopes ? NativeSharedEnvelopes{} : nativeSharedEnvelopes(track);
    const auto& envelopes = sharedEnvelopes ? *sharedEnvelopes : ownEnvelopes;
    request.noteGain.resize(static_cast<std::size_t>(frameCount), 1.0f);
    request.tension.resize(static_cast<std::size_t>(frameCount), 0.0f);
    request.breath.resize(static_cast<std::size_t>(frameCount), 0.0f);
    request.robustPitchCurve.resize(static_cast<std::size_t>(frameCount), 0.0f);
    for (int frame = 0; frame < frameCount; ++frame)
    {
        const auto time = std::min(clip.durationSeconds, static_cast<double>(frame) * framePeriodSeconds);
        for (std::size_t noteIndex = 0; noteIndex < clip.notes.size(); ++noteIndex)
        {
            const auto& note = clip.notes[noteIndex];
            const auto local = time - note.startSeconds;
            if (local < -1.0e-9 || local > note.durationSeconds + 1.0e-9) continue;
            request.formantSemitones[static_cast<std::size_t>(frame)] = note.formantSemitones;
            const auto clampedLocal = juce::jlimit(0.0, note.durationSeconds, local);
            request.noteGain[static_cast<std::size_t>(frame)] = note.gain
                * (envelopes.contains(note.id) ? backend::envelopeGainFromDb(nativeEnvelopeDbAt(*envelopes.at(note.id).points, clip.startSeconds + time))
                    : amplitudeGainAt(note.amplitudeEnvelope, clampedLocal, note.amplitudeEnvelopeBasePercent))
                * backend::nativeEnvelopeGain(note.nativeEnvelope,clampedLocal,note.durationSeconds);
            request.tension[static_cast<std::size_t>(frame)] = note.tension;
            request.breath[static_cast<std::size_t>(frame)] = note.breath;
            // Keep adjacent robust notes as separate detector regions.  A
            // boolean mask alone lets a slope limiter bridge two valid notes
            // at a hard musical interval and mistakes it for an F0 outlier.
            // Zero remains disabled; positive values identify the owning note.
            request.robustPitchCurve[static_cast<std::size_t>(frame)] =
                note.robustPitchCurve ? static_cast<float>(noteIndex + 1) : 0.0f;
            const auto cents = contourAt(note, clampedLocal);
            if (!cents) break; // Preserve the analysed unvoiced mask.
            const auto sourceCenter = note.sourceMidiCenter >= 0.0f ? note.sourceMidiCenter : note.midiNote;
            request.sourceMidi[static_cast<std::size_t>(frame)] = sourceCenter + cents->first / 100.0f;
            // Vibrato is a target-pitch edit, not a display-only decoration.
            // Keep it in the common request so NSF-HiFiGAN and every model-free
            // native backend hear exactly what the piano roll draws.
            request.targetMidi[static_cast<std::size_t>(frame)] = note.midiNote
                + (cents->second + vibratoCentsAt(note, clampedLocal)) / 100.0f;
            break;
        }
    }

    for (const auto& joinedNote : clip.notes)
    {
        if (!joinedNote.connectedToPrevious || joinedNote.nativeIndependentPitch) continue;
        const NoteData* previousNote = nullptr;
        auto previousEnd = -std::numeric_limits<double>::infinity();
        const auto joinedStart = clip.startSeconds + joinedNote.startSeconds;
        for (const auto& candidateClip : track.clips)
            for (const auto& candidate : candidateClip.notes)
            {
                const auto end = candidateClip.startSeconds + candidate.startSeconds
                    + candidate.durationSeconds;
                if (end <= joinedStart + 0.002
                    && end > previousEnd && candidate.id != joinedNote.id)
                {
                    previousEnd = end;
                    previousNote = &candidate;
                }
            }
        if (previousNote != nullptr && !previousNote->nativeIndependentPitch)
        {
            const auto previousCents = contourAt(*previousNote, previousNote->durationSeconds);
            const auto previousPitch = previousNote->midiNote + (previousCents
                ? (previousCents->second
                    + vibratoCentsAt(*previousNote, previousNote->durationSeconds)) / 100.0f
                : 0.0f);
            const auto joinSeconds = std::min(0.08,
                std::max(0.012, joinedNote.durationSeconds * 0.22));
            const auto firstFrame = juce::jlimit(0, frameCount - 1,
                static_cast<int>(std::llround(joinedNote.startSeconds / framePeriodSeconds)));
            const auto joinFrames = std::min(frameCount - firstFrame,
                std::max(2, static_cast<int>(std::ceil(joinSeconds / framePeriodSeconds))));
            if (joinFrames < 2) continue;
            for (int frame = 0; frame < joinFrames; ++frame)
            {
                auto& target = request.targetMidi[static_cast<std::size_t>(firstFrame + frame)];
                if (!(target > 0.0f)) continue;
                const auto x = static_cast<float>(frame) / static_cast<float>(joinFrames - 1);
                const auto smooth = x * x * (3.0f - 2.0f * x);
                target = previousPitch + (target - previousPitch) * smooth;
            }
        }
    }

    // Native timeline-pitch continuity: where two notes abut on the timeline
    // (not an explicit glide, just neighbours), the per-frame target otherwise
    // steps from the first note's tail pitch to the next note's head pitch in a
    // single frame -- an audible seam in the model backends.  Lay the same short
    // automatic S-transition the UTAU path uses (smoothstep across a small inset
    // either side of the boundary) directly into the target-MIDI line, so the
    // one native pitch line the models read is continuous across the seam.  Only
    // between two voiced sides, so an analysed unvoiced gap is never bridged.
    {
        std::vector<std::size_t> order(clip.notes.size());
        std::iota(order.begin(), order.end(), std::size_t { 0 });
        std::stable_sort(order.begin(), order.end(), [&](auto left, auto right)
        {
            return clip.notes[left].startSeconds < clip.notes[right].startSeconds;
        });
        const auto targetAt = [&](double seconds) -> float
        {
            const auto frame = static_cast<int>(std::llround(seconds / framePeriodSeconds));
            if (frame < 0 || frame >= frameCount) return 0.0f;
            return request.targetMidi[static_cast<std::size_t>(frame)];
        };
        for (std::size_t position = 1; position < order.size(); ++position)
        {
            const auto& previous = clip.notes[order[position - 1]];
            const auto& next = clip.notes[order[position]];
            // Measured F0 already contains the recording's own transition.
            // An analysis boundary must never introduce a target-pitch edit.
            if (nativePitchIsUnedited(previous) && nativePitchIsUnedited(next)) continue;
            // Explicit source-pitch restoration excludes artificial boundary bends.
            if (!previous.utauAutoPitchTransition || !next.utauAutoPitchTransition) continue;
            const auto previousEnd = previous.startSeconds + previous.durationSeconds;
            if (std::abs(previousEnd - next.startSeconds) > 0.002) continue;
            // An explicit connection already glided this seam above.
            if (next.connectedToPrevious) continue;
            const auto inset = automaticUtauPitchTransitionInset(
                previous.durationSeconds, next.durationSeconds);
            const auto startSeconds = previousEnd - inset;
            const auto endSeconds = next.startSeconds + inset;
            const auto startPitch = targetAt(startSeconds - framePeriodSeconds * 0.0);
            const auto endPitch = targetAt(endSeconds);
            // Both sides must be voiced (a real target); 0 marks unvoiced.
            if (!(startPitch > 0.0f) || !(endPitch > 0.0f)) continue;
            const auto firstFrame = juce::jlimit(0, frameCount - 1,
                static_cast<int>(std::llround(startSeconds / framePeriodSeconds)));
            const auto lastFrame = juce::jlimit(0, frameCount - 1,
                static_cast<int>(std::llround(endSeconds / framePeriodSeconds)));
            if (lastFrame - firstFrame < 2) continue;
            for (int frame = firstFrame; frame <= lastFrame; ++frame)
            {
                auto& target = request.targetMidi[static_cast<std::size_t>(frame)];
                if (!(target > 0.0f)) continue; // never fabricate over unvoiced
                const auto x = static_cast<float>(frame - firstFrame)
                    / static_cast<float>(lastFrame - firstFrame);
                const auto smooth = x * x * (3.0f - 2.0f * x);
                target = startPitch + (endPitch - startPitch) * smooth;
            }
        }
    }
    const auto localLines = sharedLines ? SharedPitchLines{} : sharedPitchLines(track);
    applyNativeSharedPitch(request, clip, sharedLines ? *sharedLines : localLines);
    return request;
}

// One request covering a whole glide chain: the clips' time maps stitched
// end to end into one map, and one pitch line with every seam glided so the
// single decode never meets a hard F0 step inside the phrase.
}

backend::Mld5FileRenderRequest AudioEngine::mergedRequestFor(
    const std::vector<const ClipData*>& group, const TrackData& track,
    const juce::File& hifiganModelDirectory,
    const backend::OrtExecutionConfig& inference,
    std::optional<bool> legacyUvCarrier, bool skipSharedPitch,
    const SharedPitchLines* sharedLines)
{
    const bool useLegacyUvCarrier = legacyUvCarrier.value_or(!track.nsfNoiseProtection);
    backend::Mld5FileRenderRequest request;
    request.sourceFile = group.front()->sourceFile;
    auto sourceStart = std::numeric_limits<double>::max();
    auto sourceEnd = 0.0;
    std::vector<double> targetOffsets;
    targetOffsets.reserve(group.size());
    auto targetDuration = 0.0;
    for (const auto* clip : group)
    {
        targetOffsets.push_back(targetDuration);
        targetDuration += clip->durationSeconds;
        sourceStart = std::min(sourceStart, clip->sourceOffsetSeconds);
        sourceEnd = std::max(sourceEnd, clip->sourceOffsetSeconds + clip->sourceDurationSeconds);
    }
    request.sourceOffsetSeconds = sourceStart;
    request.sourceDurationSeconds = std::max(1.0e-6, sourceEnd - sourceStart);
    request.targetDurationSeconds = targetDuration;
    request.preserveUneditedSource = std::all_of(group.begin(), group.end(), [](const auto* clip)
    {return std::all_of(clip->notes.begin(), clip->notes.end(), nativeSourcePitchIsKnown);});
    request.hifiganModelDirectory = hifiganModelDirectory;
    request.inference = inference;
    request.pitchBackend = backend::PitchRenderBackend::nsfHifigan;
    request.stretchAlgorithm = static_cast<int>(track.stretchAlgorithm);
    request.normalizeVolume = track.normalizeVolume;
    request.nsfSmoothPitchTransitions = track.nsfSmoothPitchTransitions;
    request.nsfNoiseProtection = track.nsfNoiseProtection;
    request.nsfIncrementalRender = track.nsfNoiseProtection;
    request.isGlideMerged = true;
    // The variable-hop paths read the two sides of a source discontinuity in
    // order, choosing the old source before a seam and the new one after it,
    // so their anchors must not be sorted together across the seam.
    const auto preserveSourceSeams = track.stretchAlgorithm == StretchAlgorithm::variableMelHop
        || track.stretchAlgorithm == StretchAlgorithm::nsfShiftThenSplice
        || track.stretchAlgorithm == StretchAlgorithm::hifiShifterMel
        || track.stretchAlgorithm == StretchAlgorithm::wcsNonLoop;

    for (std::size_t index = 0; index < group.size(); ++index)
    {
        const auto& clip = *group[index];
        const auto targetOffset = targetOffsets[index];
        const auto sourceOffset = clip.sourceOffsetSeconds - sourceStart;
        std::vector<backend::TimeMapPoint> localAnchors;
        if (!clip.sourceTimeMap.empty())
        {
            for (const auto& point : clip.sourceTimeMap)
                localAnchors.push_back({
                    juce::jlimit(0.0, clip.durationSeconds, point.targetSeconds),
                    juce::jlimit(0.0, clip.sourceDurationSeconds, point.sourceSeconds) });
        }
        else
        {
            for (const auto& note : clip.notes)
            {
                const auto noteStart = juce::jlimit(0.0, clip.durationSeconds, note.startSeconds);
                const auto srcStart = clip.durationSeconds > 1.0e-9
                    ? noteStart / clip.durationSeconds * clip.sourceDurationSeconds : 0.0;
                localAnchors.push_back({ noteStart, srcStart });
                if (note.consonantSeconds <= 1.0e-6 || note.attackSpeed <= 1.0e-6f) continue;
                const auto targetAttack = juce::jlimit(noteStart, clip.durationSeconds,
                    noteStart + note.consonantSeconds);
                const auto sourceAttack = juce::jlimit(srcStart, clip.sourceDurationSeconds,
                    srcStart + note.consonantSeconds * static_cast<double>(note.attackSpeed));
                localAnchors.push_back({ targetAttack, sourceAttack });
            }
        }
        localAnchors.push_back({ 0.0, 0.0 });
        localAnchors.push_back({ clip.durationSeconds, clip.sourceDurationSeconds });
        std::stable_sort(localAnchors.begin(), localAnchors.end(),
                         [](const auto& left, const auto& right)
        {
            if (std::abs(left.targetSeconds - right.targetSeconds) > 1.0e-9)
                return left.targetSeconds < right.targetSeconds;
            return left.sourceSeconds < right.sourceSeconds;
        });
        std::vector<backend::TimeMapPoint> localMap;
        for (const auto& anchor : localAnchors)
        {
            if (localMap.empty())
            {
                localMap.push_back(anchor);
                continue;
            }
            auto& previous = localMap.back();
            if (std::abs(anchor.targetSeconds - previous.targetSeconds) <= 1.0e-7)
            {
                previous.sourceSeconds = std::max(previous.sourceSeconds, anchor.sourceSeconds);
                continue;
            }
            if (anchor.sourceSeconds > previous.sourceSeconds + 1.0e-7)
                localMap.push_back(anchor);
        }
        for (const auto& anchor : localMap)
        {
            const backend::TimeMapPoint mapped {
                targetOffset + anchor.targetSeconds,
                sourceOffset + anchor.sourceSeconds
            };
            if (!request.timeMap.empty()
                && std::abs(mapped.targetSeconds - request.timeMap.back().targetSeconds) <= 1.0e-7
                && std::abs(mapped.sourceSeconds - request.timeMap.back().sourceSeconds) <= 1.0e-7)
                continue;
            request.timeMap.push_back(mapped);
        }
    }
    if (!preserveSourceSeams)
    {
        auto anchors = std::move(request.timeMap);
        request.timeMap.clear();
        std::stable_sort(anchors.begin(), anchors.end(), [](const auto& left, const auto& right)
        {
            if (std::abs(left.targetSeconds - right.targetSeconds) > 1.0e-9)
                return left.targetSeconds < right.targetSeconds;
            return left.sourceSeconds < right.sourceSeconds;
        });
        for (const auto& anchor : anchors)
        {
            if (request.timeMap.empty())
            {
                request.timeMap.push_back(anchor);
                continue;
            }
            auto& previous = request.timeMap.back();
            if (std::abs(anchor.targetSeconds - previous.targetSeconds) <= 1.0e-7)
            {
                previous.sourceSeconds = std::max(previous.sourceSeconds, anchor.sourceSeconds);
                continue;
            }
            if (anchor.sourceSeconds > previous.sourceSeconds + 1.0e-7)
                request.timeMap.push_back(anchor);
        }
    }
    if (request.timeMap.empty() || request.timeMap.back().targetSeconds < targetDuration - 1.0e-7)
        request.timeMap.push_back({ targetDuration, sourceEnd - sourceStart });

    constexpr auto framePeriodSeconds = 0.005;
    request.framePeriodMs = framePeriodSeconds * 1000.0;
    const auto frameCount = std::max(2, static_cast<int>(
        std::ceil(targetDuration / framePeriodSeconds)) + 1);
    request.sourceMidi.assign(static_cast<std::size_t>(frameCount), 0.0f);
    request.targetMidi.assign(static_cast<std::size_t>(frameCount), 0.0f);
    request.formantSemitones.assign(static_cast<std::size_t>(frameCount), 0.0f);
    const auto envelopes = nativeSharedEnvelopes(track);
    request.noteGain.assign(static_cast<std::size_t>(frameCount), 1.0f);
    request.tension.assign(static_cast<std::size_t>(frameCount), 0.0f);
    request.breath.assign(static_cast<std::size_t>(frameCount), 0.0f);
    request.robustPitchCurve.assign(static_cast<std::size_t>(frameCount), 0.0f);
    auto lastSourceMidi = 0.0f;
    auto lastTargetMidi = 0.0f;
    for (int frame = 0; frame < frameCount; ++frame)
    {
        const auto time = std::min(targetDuration, static_cast<double>(frame) * framePeriodSeconds);
        std::size_t index = 0;
        while (index + 1 < group.size() && targetOffsets[index + 1] <= time) ++index;
        const auto& clip = *group[index];
        const auto local = time - targetOffsets[index];
        for (const auto& note : clip.notes)
        {
            const auto noteLocal = local - note.startSeconds;
            if (noteLocal < -1.0e-9 || noteLocal > note.durationSeconds + 1.0e-9) continue;
            request.formantSemitones[static_cast<std::size_t>(frame)] = note.formantSemitones;
            const auto clampedLocal = juce::jlimit(0.0, note.durationSeconds, noteLocal);
            request.noteGain[static_cast<std::size_t>(frame)] = note.gain
                * (envelopes.contains(note.id) ? backend::envelopeGainFromDb(nativeEnvelopeDbAt(*envelopes.at(note.id).points, clip.startSeconds + local))
                    : amplitudeGainAt(note.amplitudeEnvelope, clampedLocal, note.amplitudeEnvelopeBasePercent))
                * backend::nativeEnvelopeGain(note.nativeEnvelope,clampedLocal,note.durationSeconds);
            request.tension[static_cast<std::size_t>(frame)] = note.tension;
            request.breath[static_cast<std::size_t>(frame)] = note.breath;
            request.robustPitchCurve[static_cast<std::size_t>(frame)] =
                note.robustPitchCurve ? static_cast<float>(index + 1) : 0.0f;
            const auto cents = contourAt(note, clampedLocal);
            // Noise protection uses a true UV mask; the unprotected legacy
            // decoder keeps its previous voiced carrier, as old projects did.
            // Explicit native unpitched/muted regions must remain zero in both.
            if (!cents)
            {
                if (useLegacyUvCarrier && !note.nativeUnpitched)
                {
                    if (lastSourceMidi > 0.0f)
                    {
                        request.sourceMidi[static_cast<std::size_t>(frame)] = lastSourceMidi;
                        request.targetMidi[static_cast<std::size_t>(frame)] = lastTargetMidi;
                    }
                    continue;
                }
                break;
            }
            const auto sourceCenter = note.sourceMidiCenter >= 0.0f
                ? note.sourceMidiCenter : note.midiNote;
            lastSourceMidi = sourceCenter + cents->first / 100.0f;
            lastTargetMidi = note.midiNote
                + (cents->second + vibratoCentsAt(note, clampedLocal)) / 100.0f;
            request.sourceMidi[static_cast<std::size_t>(frame)] = lastSourceMidi;
            request.targetMidi[static_cast<std::size_t>(frame)] = lastTargetMidi;
            break;
        }
    }
    // Glide each seam out of the previous clip's tail pitch, so the one decode
    // never sees the step that splicing two independent renders would leave.
    for (std::size_t index = 1; index < group.size(); ++index)
    {
        const auto& previousClip = *group[index - 1];
        const auto& nextClip = *group[index];
        if ((!previousClip.notes.empty() && previousClip.notes.back().nativeIndependentPitch)
            || (!nextClip.notes.empty() && nextClip.notes.front().nativeIndependentPitch)) continue;
        auto previousPitch = previousClip.notes.empty()
            ? 0.0 : static_cast<double>(previousClip.notes.back().midiNote);
        if (!previousClip.notes.empty())
        {
            const auto& note = previousClip.notes.back();
            const auto cents = contourAt(note, note.durationSeconds);
            previousPitch = note.midiNote + (cents
                ? (cents->second + vibratoCentsAt(note, note.durationSeconds)) / 100.0f
                : 0.0f);
        }
        const auto joinSeconds = std::min(0.08, std::max(0.012, nextClip.durationSeconds * 0.22));
        const auto firstFrame = juce::jlimit(0, frameCount - 1,
            static_cast<int>(std::llround(targetOffsets[index] / framePeriodSeconds)));
        const auto joinFrames = std::min(frameCount - firstFrame,
            std::max(2, static_cast<int>(std::ceil(joinSeconds / framePeriodSeconds))));
        if (joinFrames < 2) continue;
        for (int frame = 0; frame < joinFrames; ++frame)
        {
            auto& target = request.targetMidi[static_cast<std::size_t>(firstFrame + frame)];
            if (!(target > 0.0f)) continue;
            const auto x = static_cast<float>(frame) / static_cast<float>(joinFrames - 1);
            const auto smooth = x * x * (3.0f - 2.0f * x);
            target = static_cast<float>(previousPitch + (target - previousPitch) * smooth);
        }
    }
    if (!skipSharedPitch)
    {
        const auto localLines = sharedLines ? SharedPitchLines{} : sharedPitchLines(track);
        const auto& lines = sharedLines ? *sharedLines : localLines;
        for (std::size_t index = 0; index < group.size(); ++index)
            applyNativeSharedPitch(request, *group[index], lines, targetOffsets[index]);
    }
    return request;
}

namespace
{
// Everything about one note that decides how it is rendered.  Written by
// the cache key, and hashed by the waveform display to ask whether the note
// under a drawn waveform is still the note that produced it -- one list, so
// the two can never disagree about what counts as an edit.
void writeNoteRenderFields(juce::MemoryOutputStream& stream, const NoteData& note,
                           bool withAmplitudeEnvelope = true)
{
    // Optional semantic intent: legacy false writes NO new bytes, preserving
    // accepted project/cache keys exactly. True changes the playable domain.
    if(note.pitchSupportAuthored)stream.writeString("pitch-support-authored-v1");
    const auto label = note.label.toUTF8();
    stream.write(label.getAddress(), label.sizeInBytes());
    const auto flags = note.utauFlags.toUTF8();
    stream.write(flags.getAddress(), flags.sizeInBytes());
    stream.writeInt(note.utauConsonantVelocity);
    stream.writeBool(note.utauPreutteranceOverrideEnabled);
    stream.writeDouble(note.utauPreutteranceSeconds);
    stream.writeBool(note.utauOverlapOverrideEnabled);
    stream.writeDouble(note.utauOverlapSeconds);
    stream.writeDouble(note.utauStpSeconds);
    stream.writeDouble(note.utauModulationPercent);
    // A note's own oto decides what it sounds, so it is part of what counts as
    // an edit -- for the render cache and for the waveform drawn under it.
    stream.writeBool(note.utauOto.enabled);
    stream.writeDouble(note.utauOto.offsetMs);
    stream.writeDouble(note.utauOto.consonantMs);
    stream.writeDouble(note.utauOto.cutoffMs);
    stream.writeDouble(note.utauOto.preutteranceMs);
    stream.writeDouble(note.utauOto.overlapMs);
    stream.writeDouble(note.utauOto.onsetMs);
    stream.writeDouble(note.utauOto.glideMs);
    stream.writeDouble(note.utauOto.nucleusMs);
    stream.writeBool(note.utauOto.hasRegions);
    stream.writeString(note.utauOto.classes);
    stream.writeBool(note.vibratoEnabled);
    stream.writeDouble(note.vibratoLengthPercent);
    stream.writeDouble(note.vibratoCycleMs);
    stream.writeDouble(note.vibratoDepthCents);
    stream.writeDouble(note.vibratoFadeInPercent);
    stream.writeDouble(note.vibratoFadeOutPercent);
    stream.writeDouble(note.vibratoPhasePercent);
    stream.writeDouble(note.vibratoOffsetPercent);
    stream.writeDouble(note.vibratoEndPercent);
    stream.writeDouble(note.vibratoReferenceDurationSeconds);
    stream.writeDouble(note.vibratoTimeOffsetSeconds);
    stream.writeBool(note.utauFlagSplit);
    stream.writeBool(note.utauFlagCurveEnabled);
    for (const auto& curve : note.utauFlagCurves)
    {
        if (curve.flag.startsWith("DS:AUTO:")) continue;
        stream.writeString(curve.flag);
        for (const auto& point : curve.points)
        {
            stream.writeDouble(point.timeSeconds);
            stream.writeDouble(point.value);
            stream.writeInt(static_cast<int>(point.shape));
            for (const auto handle : { point.bezierX1, point.bezierY1,
                                       point.bezierX2, point.bezierY2 })
                stream.writeFloat(handle);
        }
    }
    stream.writeBool(note.utauSplice);
    stream.writeBool(note.utauAutoPitchTransition);
    stream.writeBool(note.nativeIndependentPitch);
    stream.writeBool(note.nativeUnpitched);
    stream.writeBool(note.nativePitchHandlesPlaced);
    stream.writeBool(note.nativeSparsePitchControls);
    for (const auto& text : { note.utauRegionFlags1, note.utauRegionFlags2,
                              note.utauRegionFlags3, note.utauRegionFlags4 })
    {
        const auto raw = text.toUTF8();
        stream.write(raw.getAddress(), raw.sizeInBytes());
    }
    stream.writeBool(note.utauJieSplitSet);
    if(!note.cvvcBoundaries.empty()){stream.writeString("cvvc-boundaries-v1");
        stream.writeString(juce::JSON::toString(backend::cvvcBoundariesToVar(note.cvvcBoundaries),true,17));}
    stream.writeDouble(note.utauJieSplit1);
    stream.writeDouble(note.utauJieSplit2);
    stream.writeDouble(note.utauJieSplit3);
    stream.writeDouble(note.startSeconds);
    stream.writeDouble(note.durationSeconds);
    stream.writeFloat(note.midiNote);
    stream.writeFloat(note.sourceMidiCenter);
    stream.writeBool(note.sourcePitchMeasured);
    stream.writeString(note.diffSingerTiming);
    stream.writeString(note.diffSingerPronunciation);
    stream.writeDouble(note.consonantSeconds);
    stream.writeFloat(note.attackSpeed);
    stream.writeByte(static_cast<char>(note.robustPitchCurve ? 1 : 0));
    stream.writeByte(static_cast<char>(note.connectedToPrevious ? 1 : 0));
    stream.writeByte(static_cast<char>(note.connectedToNext ? 1 : 0));
    stream.writeFloat(note.modulation);
    stream.writeFloat(note.drift);
    stream.writeFloat(note.tension);
    stream.writeFloat(note.breath);
    stream.writeFloat(note.formantSemitones);
    stream.writeFloat(note.gain);
    if (withAmplitudeEnvelope)
    {
        stream.writeFloat(note.amplitudeEnvelopeBasePercent);
        stream.writeString(juce::JSON::toString(backend::nativeEnvelopeToVar(note.nativeEnvelope),true,17));
        stream.writeInt(note.utauTailFadeMode);
        stream.writeString(backend::mixedEnvelopeText(note.utauTailFade));
        stream.writeString(backend::mixedEnvelopeText(note.utauTailFade.head));
        if(note.utauTailFadeMode!=0)stream.writeString("oto-single-tail-v2");
        stream.writeDouble(note.utauTailFade.startFraction);
        stream.writeDouble(note.utauTailFade.endFraction);
        stream.writeDouble(note.utauTailFade.startGain);
        stream.writeDouble(note.utauTailFade.endGain);
        stream.writeDouble(note.utauTailFade.curvePower);
        stream.writeInt(note.utauTailFade.head.mode);
        stream.writeDouble(note.utauTailFade.head.startFraction);
        stream.writeDouble(note.utauTailFade.head.endFraction);
        stream.writeDouble(note.utauTailFade.head.startGain);
        stream.writeDouble(note.utauTailFade.head.endGain);
        stream.writeDouble(note.utauTailFade.head.curvePower);
        stream.writeBool(note.utauTailFade.head.customCurve);
        stream.writeDouble(note.utauTailFade.head.control1Time);
        stream.writeDouble(note.utauTailFade.head.control1Progress);
        stream.writeDouble(note.utauTailFade.head.control2Time);
        stream.writeDouble(note.utauTailFade.head.control2Progress);
    stream.writeBool(note.utauTailFade.customCurve);
    stream.writeDouble(note.utauTailFade.control1Time);
    stream.writeDouble(note.utauTailFade.control1Progress);
    stream.writeDouble(note.utauTailFade.control2Time);
    stream.writeDouble(note.utauTailFade.control2Progress);
        stream.writeInt64(static_cast<juce::int64>(note.amplitudeEnvelope.size()));
        for (const auto& point : note.amplitudeEnvelope)
        {
            stream.writeDouble(point.timeSeconds);
            stream.writeFloat(point.gainDb);
            stream.writeBool(point.linearToNext);
            stream.writeBool(point.nativeSeamAnchor);
        }
    }
    for (const auto& [part, curves] : note.cvvcEnvelopes)
        for (const auto& [key, points] : curves)
        {
            if (key.isEmpty() && !withAmplitudeEnvelope) continue;
            stream.writeString("cvvc-envelope-v1"); stream.writeInt(part); stream.writeString(key);
            stream.writeInt(static_cast<int>(points.size()));
            for (const auto& point : points)
            {
                stream.writeDouble(point.position); stream.writeFloat(point.value); stream.writeBool(point.linear);
                stream.writeBool(point.bezier.has_value());
                if(point.bezier){stream.writeDouble(point.bezier->control1Time);stream.writeDouble(point.bezier->control1Progress);
                    stream.writeDouble(point.bezier->control2Time);stream.writeDouble(point.bezier->control2Progress);}
            }
        }
    for (const auto& point : note.contour)
    {
        stream.writeDouble(point.timeSeconds);
        stream.writeFloat(point.relativeCents);
        stream.writeFloat(point.withoutVibratoCents);
        stream.writeByte(static_cast<char>(point.voiced ? 1 : 0));
        stream.writeFloat(point.manualTargetCents);
        stream.writeByte(static_cast<char>(point.hasManualTarget ? 1 : 0));
    }
    if (!note.diffSingerPitchOffset.empty())
    {
        stream.writeString("DS pitch offset");
        stream.writeInt64(static_cast<juce::int64>(note.diffSingerPitchOffset.size()));
        for (const auto& point : note.diffSingerPitchOffset)
        { stream.writeDouble(point.timeSeconds);stream.writeFloat(point.targetMidi); }
    }
    stream.writeInt64(static_cast<juce::int64>(note.pitchControlPoints.size()));
    for (const auto& point : note.pitchControlPoints)
    {
        stream.writeDouble(point.timeSeconds);
        stream.writeFloat(point.targetMidi);
        stream.writeInt(static_cast<int>(point.shape));
        stream.writeFloat(point.bezierX1);
        stream.writeFloat(point.bezierY1);
        stream.writeFloat(point.bezierX2);
        stream.writeFloat(point.bezierY2);
        if (point.highPrecisionBezier) stream.writeString("cropped-bezier-double-v1");
    }
}

// The note as the cache key sees it, in one number.  Two notes with the same
// value render the same audio, which is what lets a drawn waveform be checked
// against the note still under it.
std::uint64_t noteRenderHash(const NoteData& note)
{
    return AudioEngine::utauNoteRenderHash(note);
}

std::uint64_t noteRenderHashImpl(const NoteData& note, bool withAmplitudeEnvelope = true)
{
    juce::MemoryOutputStream stream;
    writeNoteRenderFields(stream, note, withAmplitudeEnvelope);
    // FNV-1a: no dependency, and collisions here cost a waveform that is drawn
    // when it should not be, not audio that is wrong.
    std::uint64_t hash = 1469598103934665603ull;
    const auto* bytes = static_cast<const unsigned char*>(stream.getData());
    for (std::size_t index = 0; index < stream.getDataSize(); ++index)
    {
        hash ^= bytes[index];
        hash *= 1099511628211ull;
    }
    return hash;
}

std::string voicebankRenderStamp(const TrackData& track)
{
    juce::MemoryOutputStream stream;
    if (trackUsesVoicebankSynthesis(track) && track.voicebankDirectory.isDirectory())
    {
        juce::Array<juce::File> otoFiles;
        track.voicebankDirectory.findChildFiles(
            otoFiles, juce::File::findFiles, true, "*");
        otoFiles.removeIf([](const juce::File& file)
        {
            const auto name = file.getFileName();
            return !name.equalsIgnoreCase("oto.ini")
                && !name.equalsIgnoreCase("oto.jie.ini")
                && !name.equalsIgnoreCase("oto4.ini")
                && !name.equalsIgnoreCase("otomou.ini")
                && !name.equalsIgnoreCase("presamp.ini")
                && !name.equalsIgnoreCase("prefix.map")
                && !file.hasFileExtension("yaml;json;onnx;emb;txt;wav;flac;aif;aiff");
        });
        otoFiles.sort();
        for (const auto& file : otoFiles)
        {
            const auto relative = file.getRelativePathFrom(
                track.voicebankDirectory).toUTF8();
            stream.write(relative.getAddress(), relative.sizeInBytes());
            stream.writeInt64(file.getLastModificationTime().toMilliseconds());
            stream.writeInt64(file.getSize());
        }
    }
    return std::string(static_cast<const char*>(stream.getData()), stream.getDataSize());
}

void writeNativePitchContext(juce::MemoryOutputStream& stream, const ClipData& clip, const TrackData& track)
{
    if (!trackShowsAllNativeRegions(track)) return;
    stream.writeString("native-shared-pitch-envelope-131");
    // Cross-recording author points can change this clip's target F0. Cache the
    // resolved line used by its notes, including the neighbouring point spans.
    stream.writeString("native-point-first-241");
    const auto lines = sharedPitchLines(track);
    std::set<const SharedPitchLine*> written;
    for (const auto& note : clip.notes)
        if (const auto* member = lines.memberFor(note.id); member && member->renderSharedCurve && written.insert(member->line.get()).second)
            for (const auto& piece : member->line->pieces)
            {
                stream.writeDouble(piece.from); stream.writeInt64(static_cast<juce::int64>(piece.points.size()));
                for (const auto& point : piece.points)
                {
                    stream.writeDouble(point.timeSeconds); stream.writeFloat(point.targetMidi); stream.writeInt(static_cast<int>(point.shape));
                    stream.writeFloat(point.bezierX1); stream.writeFloat(point.bezierY1); stream.writeFloat(point.bezierX2); stream.writeFloat(point.bezierY2);
                }
            }
    const auto group = clip.nativePitchGroupId.isNotEmpty() ? clip.nativePitchGroupId : clip.id;
    std::vector<NoteData> notes;
    for (const auto& parent : track.clips)
        for (const auto& part : expandedClipParts(parent))
        {
            const auto otherGroup = part.nativePitchGroupId.isNotEmpty() ? part.nativePitchGroupId
                : parent.nativeAudioLinked ? parent.id : part.id;
            if (otherGroup != group) continue;
            for (auto note : part.notes)
            {
                note.startSeconds += part.startSeconds;
                notes.push_back(std::move(note));
            }
        }
    std::stable_sort(notes.begin(), notes.end(), [](const auto& a, const auto& b)
    { return a.startSeconds == b.startSeconds ? a.id < b.id : a.startSeconds < b.startSeconds; });
    stream.writeInt64(static_cast<juce::int64>(notes.size()));
    for (const auto& note : notes) writeNoteRenderFields(stream, note);
}

backend::RequestedVoicebankFamily requestedVoicebankFamily(const TrackData& track)
{
    if (!trackUsesVoicebankSynthesis(track)) return backend::RequestedVoicebankFamily::legacyAuto;
    return trackIsDiffSinger(track) ? backend::RequestedVoicebankFamily::diffSinger
        : backend::RequestedVoicebankFamily::utau;
}

std::string renderKey(const ClipData& clip, const TrackData& track,
                      const juce::File& hifiganModelDirectory,
                      const backend::OrtExecutionConfig& inference,
                      const juce::File& utauResamplerFile, const juce::var& dsOptions = {},
                      UtauOutputEngine defaultEngine = UtauOutputEngine::resampler,
                      const juce::File& wavtool = {}, const std::string* voicebankStamp = nullptr,
                      const std::vector<NativeConnection>* connections = nullptr)
{
    juce::MemoryOutputStream stream;
    writeNativePitchContext(stream, clip, track);
    if (connections != nullptr && !connections->empty())
    {
        const auto digest = connectionPitchDigest(track, *connections, clip);
        // Empty graphs keep the historical cache bytes exactly unchanged.
        if (digest.isNotEmpty()) { stream.writeString("explicit-pitch-bridge-v1"); stream.writeString(digest); }
    }
    if (!trackUsesVoicebankSynthesis(track) && !trackIsDiffSinger(track))
        stream.writeInt(2026100711); // Free-crop sample alignment and source-preservation semantics.
    if (trackIsDiffSinger(track)) stream.writeString(juce::JSON::toString(dsOptions));
    if (trackUsesVoicebankSynthesis(track))
    {
        backend::UtauRenderRequest routing;
        routing.voicebankDirectory = track.voicebankDirectory;
        routing.requestedFamily = requestedVoicebankFamily(track);
        const auto family = backend::resolveRequestedVoicebankFamily(routing);
        // Matching historic routes (including native Hachi files) add no bytes.
        // Only an explicitly corrected route/blocked old fallback gets a new key.
        if (family.cacheDiscriminator.isNotEmpty()) stream.writeString(family.cacheDiscriminator);
    }
    // The render order reaches the per-clip render through matchNsfSourceLevel,
    // so two orders are two different buffers and must not share a cache entry.
    stream.writeInt(static_cast<int>(track.renderOrder));
    stream.writeBool(track.mergeContiguousSource);
    if (trackUsesVoicebankSynthesis(track))
    {
        // A selected note can read a curve edited on an unselected neighbour
        // or another clip. The clip cache must include those dependencies too.
        stream.writeInt(20260921);
        stream.writeDouble(clip.startSeconds);
        stream.writeInt64(static_cast<juce::int64>(track.clips.size()));
        for (const auto& neighbourClip : track.clips)
        {
            stream.writeDouble(neighbourClip.startSeconds);
            stream.writeInt64(static_cast<juce::int64>(neighbourClip.notes.size()));
            for (const auto& neighbour : neighbourClip.notes)
                writeNoteRenderFields(stream, neighbour);
        }
    }
    const auto path = clip.sourceFile.getFullPathName().toUTF8();
    stream.write(path.getAddress(), path.sizeInBytes());
    stream.writeInt64(clip.sourceFile.getLastModificationTime().toMilliseconds());
    stream.writeDouble(clip.sourceOffsetSeconds);
    stream.writeDouble(clip.sourceDurationSeconds);
    stream.writeDouble(clip.durationSeconds);
    stream.writeInt64(static_cast<juce::int64>(clip.sourceTimeMap.size()));
    for (const auto& point : clip.sourceTimeMap)
    {
        stream.writeDouble(point.targetSeconds);
        stream.writeDouble(point.sourceSeconds);
    }
    stream.writeInt(static_cast<int>(track.pitchAlgorithm));
    stream.writeInt(static_cast<int>(track.stretchAlgorithm));
    stream.writeBool(track.normalizeVolume);
    stream.writeBool(track.nsfSmoothPitchTransitions);
    stream.writeBool(track.nsfNoiseProtection);
    stream.writeBool(track.nativeNsfAudio);
    stream.writeInt(static_cast<int>(track.utauMode));
    stream.writeBool(track.chineseCvvc);
    stream.writeBool(track.utauPianoPreview);
    stream.writeBool(track.cvvcCurveCrossfade);
    if(track.chineseCvvc)stream.writeString("cvvc-single-edge-envelope-v2-selection-warnings-v1");
    if(trackUsesVoicebankSynthesis(track)&&!trackIsDiffSinger(track))
        stream.writeString("utau-point-first-pitch-v2");
    if (track.chineseCvvc && track.utauMode == UtauMode::classic)
    {
        stream.writeString("zh-cvvc-signed-velocity-timing-v4");
        stream.writeString("zh-cvvc-shared-timeline-pitch-v1");
    }
    stream.writeInt(static_cast<int>(effectiveUtauOutputEngine(track, defaultEngine)));
    if (effectiveUtauOutputEngine(track, defaultEngine) == UtauOutputEngine::pcNsfHifigan)
        stream.writeString("hifisampler-held-crossfade-v2");
    const auto selectedResampler = effectiveUtauResampler(track, utauResamplerFile);
    const auto selectedWavtool = effectiveUtauWavtool(track, wavtool);
    stream.writeString(selectedWavtool.getFullPathName());
    stream.writeInt64(selectedWavtool.getLastModificationTime().toMilliseconds());
    stream.writeInt64(selectedWavtool.getSize());
    stream.writeInt(track.utauConsonantVelocity);
    const auto globalFlags = track.utauGlobalFlags.toUTF8();
    stream.write(globalFlags.getAddress(), globalFlags.sizeInBytes());
    const auto voicebankPath = track.voicebankDirectory.getFullPathName().toUTF8();
    stream.write(voicebankPath.getAddress(), voicebankPath.sizeInBytes());
    stream.writeInt64(track.voicebankDirectory.getLastModificationTime().toMilliseconds());
    stream.writeString(track.diffSingerLanguage);
    stream.writeString(track.diffSingerSpeaker);
    stream.writeString(track.diffSingerDictionary);
    const auto bankStamp = voicebankStamp != nullptr ? *voicebankStamp : voicebankRenderStamp(track);
    stream.write(bankStamp.data(), bankStamp.size());
    const auto resamplerPath = selectedResampler.getFullPathName().toUTF8();
    stream.write(resamplerPath.getAddress(), resamplerPath.sizeInBytes());
    stream.writeInt64(selectedResampler.getLastModificationTime().toMilliseconds());
    stream.writeInt64(selectedResampler.getSize());
    const auto modelPath = hifiganModelDirectory.getFullPathName().toUTF8();
    stream.write(modelPath.getAddress(), modelPath.sizeInBytes());
    auto modelDirectory = hifiganModelDirectory.existsAsFile()
        ? hifiganModelDirectory.getParentDirectory() : hifiganModelDirectory;
    if (!modelDirectory.getChildFile("pc_nsf_hifigan.onnx").existsAsFile()) {
        const auto env = juce::SystemStats::getEnvironmentVariable("HACHISHIFTER_NSF_HIFIGAN_MODEL_DIR", {});
        if (env.isNotEmpty()) modelDirectory = juce::File(env);
        else if (hifiganModelDirectory == juce::File{}) modelDirectory = juce::File::getSpecialLocation(
            juce::File::currentExecutableFile).getParentDirectory().getChildFile("models/nsf_hifigan");
    }
    const auto model = modelDirectory.getChildFile("pc_nsf_hifigan.onnx");
    const auto config = modelDirectory.getChildFile("config.json");
    stream.writeInt64(model.getLastModificationTime().toMilliseconds());
    stream.writeInt64(model.getSize());
    stream.writeInt64(config.getLastModificationTime().toMilliseconds());
    stream.writeInt64(config.getSize());
    for (const auto* relative : { "hifisampler.json", "hnsep/model.onnx", "hnsep/config.json" }) {
        const auto file = modelDirectory.getChildFile(relative);
        stream.writeInt64(file.getLastModificationTime().toMilliseconds());
        stream.writeInt64(file.getSize());
    }
    stream.writeInt(static_cast<int>(inference.requested));
    stream.writeInt(inference.deviceIndex);
    stream.writeInt(inference.intraOpThreads);
    for (const auto& note : clip.notes)
        writeNoteRenderFields(stream, note);
    return std::string(static_cast<const char*>(stream.getData()), stream.getDataSize());
}

// One cache entry per phrase: the clips it is made of, in order, plus the
// order itself so the two never collide.
std::string mergedRenderKey(const std::vector<const ClipData*>& group, const TrackData& track,
                            const juce::File& hifiganModelDirectory,
                            const backend::OrtExecutionConfig& inference,
                            const std::vector<NativeConnection>* connections = nullptr)
{
    std::string key = "merged|" + std::to_string(static_cast<int>(track.renderOrder)) + "|";
    for (const auto* clip : group)
        key += renderKey(*clip, track, hifiganModelDirectory, inference, {}, {},
            UtauOutputEngine::resampler, {}, nullptr, connections) + ";";
    return key;
}

// One musical planner shared by UI/native projections and host packets. The
// optional switches retain explicit bench probes, never saved rendering modes.
std::vector<AudioEngine::ProjectedNotePitch> projectTrackPitchImpl(const TrackData& track,
    const std::vector<NativeConnection>& connections,bool ownHostPitch=true,bool nominalHostBridge=true)
{
    // Read over the whole track, not this request: a render is of the notes
    // selected, and the point that bends a selected note's tail may belong to
    // a note that was not.
    const auto sharedLines = ownHostPitch ? SharedPitchLines{} : sharedPitchLines(track);
    const auto pointLines=trackUsesVoicebankSynthesis(track)&&!trackIsDiffSinger(track)&&ownHostPitch
        ?utauPointCurves(track):std::map<juce::String,std::shared_ptr<const UtauPointCurve>>{};
    // A note's pitch before any transition is laid over it: the curve it is
    // sung along, and where its own line runs from and to -- which is where
    // the roll hands it over to the transition: its own outermost points when
    // it has points of its own, and the inset the automatic curve is drawn
    // with when it has none.
    struct PlannedPitch
    {
        backend::UtauNoteRenderSpec spec;   // its pitch only; startSeconds absolute
        bool automaticTransition = false;
        bool sharesPrevious = false;
        bool sharesNext = false;
        bool ownPointsPlaced = false;
        double ownFirst = 0.0;
        double ownLast = 0.0;
    };
    const auto planPitch = [&sharedLines, &pointLines, ownHostPitch, nominalHostBridge,
        refineUtau=trackUsesVoicebankSynthesis(track)&&!trackIsDiffSinger(track)](const NoteData& note, double absoluteStart)
    {
        PlannedPitch planned;
        auto& renderedNote = planned.spec;
        renderedNote.startSeconds = absoluteStart;
        renderedNote.durationSeconds = note.durationSeconds;
        renderedNote.midiNote = note.midiNote;
        planned.automaticTransition = note.utauAutoPitchTransition;
        const auto* sharedMember = ownHostPitch ? nullptr : sharedLines.memberFor(note.id);
        planned.sharesPrevious = sharedMember != nullptr && sharedMember->joinsPrevious;
        planned.sharesNext = sharedMember != nullptr && sharedMember->joinsNext;
        if(const auto found=pointLines.find(note.id);found!=pointLines.end())
        {
            const auto line=found->second;
            planned.sharesPrevious=planned.sharesNext=true;
            auto first=-.6,last=note.durationSeconds+.3;
            for(const auto& p:note.pitchControlPoints){first=std::min(first,p.timeSeconds);last=std::max(last,p.timeSeconds);}
            const auto cents=[line,absoluteStart,note](double t)
            {return (line->midiAt(absoluteStart+t)-note.midiNote)*100.f
                +float(vibratoCentsAt(note,juce::jlimit(0.,note.durationSeconds,t)));};
            std::vector<double> knots{first,last};
            for(auto t=first;t<last;t+=.005)knots.push_back(t);
            for(const auto& p:line->points)if(p.timeSeconds>=absoluteStart+first&&p.timeSeconds<=absoluteStart+last)knots.push_back(p.timeSeconds-absoluteStart);
            std::sort(knots.begin(),knots.end());knots.erase(std::unique(knots.begin(),knots.end()),knots.end());
            renderedNote.pitchCurve.push_back({knots.front(),cents(knots.front())});
            std::function<void(double,float,double,float,int)> refine;
            refine=[&](double a,float va,double b,float vb,int depth)
            {
                bool accurate=true;for(const auto f:{.25,.5,.75})if(std::abs(cents(a+(b-a)*f)-(va+(vb-va)*float(f)))>.1f){accurate=false;break;}
                if(!accurate&&depth<20&&b-a>1.e-6){const auto m=(a+b)*.5;const auto v=cents(m);refine(a,va,m,v,depth+1);refine(m,v,b,vb,depth+1);}
                else renderedNote.pitchCurve.push_back({b,vb});
            };
            for(size_t i=1;i<knots.size();++i)refine(knots[i-1],cents(knots[i-1]),knots[i],cents(knots[i]),0);
            renderedNote.timelinePitchCents=cents;
        }
        else if (sharedMember != nullptr)
        {
            // The renderer queries the immutable line at its actual sounding
            // times. The sampled vector below is only a preview/test view, not
            // a limit on the lead-in or tail that PIT is allowed to read.
            const auto line = sharedMember->line;
            const auto centsAt = [line, absoluteStart, note](double time)
            {
                return (line->midiAt(absoluteStart + time) - note.midiNote) * 100.0f
                    + static_cast<float>(vibratoCentsAt(note,
                          juce::jlimit(0.0, note.durationSeconds, time)));
            };
            renderedNote.timelinePitchCents = centsAt;
            constexpr auto before = 0.6;
            constexpr auto after = 0.3;
            const auto last = note.durationSeconds + after;
            // Every five milliseconds, on the grid a note's own curve has always
            // been sampled on -- from its first point, stepping exactly as it
            // always stepped -- so inside its own stretch a note is sent the
            // very same samples as before; the grid is only carried further,
            // back over its consonant and on past its end.
            const auto origin = note.pitchControlPoints.empty()
                ? 0.0 : std::min(0.0, note.pitchControlPoints.front().timeSeconds);
            std::vector<double> times;
            times.reserve(static_cast<std::size_t>(std::ceil((last + before) / 0.005)) + 16);
            for (auto time = origin - 0.005; time >= -before; time -= 0.005) times.push_back(time);
            for (auto time = origin; time < note.durationSeconds; time += 0.005) times.push_back(time);
            times.push_back(note.durationSeconds);
            for (auto time = note.durationSeconds + 0.005; time < last; time += 0.005)
                times.push_back(time);
            times.push_back(last);
            for (const auto corner : line->cornersBetween(absoluteStart - before, absoluteStart + last))
                times.push_back(corner - absoluteStart);
            std::sort(times.begin(), times.end());
            times.erase(std::unique(times.begin(), times.end(), [](double left, double right)
            {
                return std::abs(left - right) < 1.0e-7;
            }), times.end());
            renderedNote.pitchCurve.reserve(times.size());
            for (const auto time : times)
                renderedNote.pitchCurve.push_back({ time, centsAt(time) });
        }
        else if (!note.pitchControlPoints.empty())
        {
            auto firstTime = std::min(0.0,
                note.pitchControlPoints.front().timeSeconds);
            // Legacy host support remains nominal even when imported control
            // geometry extends past the body. Only an actual authoring action
            // opts into its stored curve extent; coordinates remain canonical.
            auto lastTime = note.durationSeconds;
            if (!ownHostPitch || note.pitchSupportAuthored)
                for (const auto& point : note.pitchControlPoints)
                {
                    lastTime = std::max(lastTime, point.timeSeconds);
                    if(note.pitchSupportAuthored)firstTime=std::min(firstTime,point.timeSeconds);
                }
            renderedNote.pitchCurve.reserve(static_cast<std::size_t>(
                std::ceil((lastTime - firstTime) / 0.005)) + 2);
            const auto centsAt = [note](double time)
            {
                return (evaluatePitchCurve(note.pitchControlPoints, time) - note.midiNote)
                    * 100.0f + static_cast<float>(vibratoCentsAt(note,
                        juce::jlimit(0.0, note.durationSeconds, time)));
            };
            renderedNote.timelinePitchCents = centsAt;
            for (auto time = firstTime; time < lastTime; time += 0.005)
                renderedNote.pitchCurve.push_back({ time, centsAt(time) });
            renderedNote.pitchCurve.push_back({ lastTime, centsAt(lastTime) });
            if(refineUtau)
            {
            for(const auto& point:note.pitchControlPoints)
                if(point.timeSeconds>=firstTime&&point.timeSeconds<=lastTime)
                    renderedNote.pitchCurve.push_back({point.timeSeconds,centsAt(point.timeSeconds)});
            std::stable_sort(renderedNote.pitchCurve.begin(),renderedNote.pitchCurve.end(),[](const auto& a,const auto& b){return a.timeSeconds<b.timeSeconds;});
            renderedNote.pitchCurve.erase(std::unique(renderedNote.pitchCurve.begin(),renderedNote.pitchCurve.end(),[](const auto& a,const auto& b){return a.timeSeconds==b.timeSeconds;}),renderedNote.pitchCurve.end());
            const auto coarse=std::move(renderedNote.pitchCurve);
            renderedNote.pitchCurve={coarse.front()};
            std::function<void(double,float,double,float,int)> refine;
            refine=[&](double a,float va,double b,float vb,int depth)
            {
                bool accurate=true;
                for(const auto fraction:{.25,.5,.75})
                    if(std::abs(centsAt(a+(b-a)*fraction)-(va+(vb-va)*static_cast<float>(fraction)))>.1f){accurate=false;break;}
                if(!accurate&&depth<20&&b-a>1.e-6)
                {const auto mid=(a+b)*.5;const auto value=centsAt(mid);refine(a,va,mid,value,depth+1);refine(mid,value,b,vb,depth+1);}
                else renderedNote.pitchCurve.push_back({b,vb});
            };
            for(size_t i=1;i<coarse.size();++i)refine(coarse[i-1].timeSeconds,coarse[i-1].cents,coarse[i].timeSeconds,coarse[i].cents,0);
            }
            if(note.pitchSupportAuthored)
            {
                // Authoring guards can be shorter than the legacy 5ms grid.
                // Retain actual out-of-body corners and body boundaries without
                // changing any legacy-false sample or inventing support points.
                for(const auto& point:note.pitchControlPoints)
                    if(point.timeSeconds<0||point.timeSeconds>note.durationSeconds)
                        renderedNote.pitchCurve.push_back({point.timeSeconds,centsAt(point.timeSeconds)});
                renderedNote.pitchCurve.push_back({0,centsAt(0)});
                renderedNote.pitchCurve.push_back({note.durationSeconds,centsAt(note.durationSeconds)});
                std::stable_sort(renderedNote.pitchCurve.begin(),renderedNote.pitchCurve.end(),[](const auto& a,const auto& b){return a.timeSeconds<b.timeSeconds;});
                renderedNote.pitchCurve.erase(std::unique(renderedNote.pitchCurve.begin(),renderedNote.pitchCurve.end(),[](const auto& a,const auto& b){return a.timeSeconds==b.timeSeconds;}),renderedNote.pitchCurve.end());
            }
        }
        else
        {
            if (note.vibratoEnabled)
            {
                // A plain UTAU note carries a two-point contour: its start and
                // its end.  Sampling the swing only at those two instants
                // flattens it away entirely, so resample the base pitch densely
                // and add the swing to every point.
                const auto baseCentsAt = [&note](double time)
                {
                    if (note.contour.empty()) return 0.0f;
                    if (time <= note.contour.front().timeSeconds)
                        return renderedPitchCents(note, note.contour.front());
                    if (time >= note.contour.back().timeSeconds)
                        return renderedPitchCents(note, note.contour.back());
                    for (std::size_t index = 1; index < note.contour.size(); ++index)
                    {
                        const auto& left = note.contour[index - 1];
                        const auto& right = note.contour[index];
                        if (time > right.timeSeconds) continue;
                        const auto width = right.timeSeconds - left.timeSeconds;
                        const auto amount = width > 1.0e-9
                            ? static_cast<float>((time - left.timeSeconds) / width) : 0.0f;
                        return renderedPitchCents(note, left)
                            + (renderedPitchCents(note, right)
                               - renderedPitchCents(note, left)) * amount;
                    }
                    return renderedPitchCents(note, note.contour.back());
                };
                auto firstTime=0.0,lastTime=note.durationSeconds;
                if(note.pitchSupportAuthored)for(const auto& point:note.contour)
                {firstTime=std::min(firstTime,point.timeSeconds);lastTime=std::max(lastTime,point.timeSeconds);}
                renderedNote.pitchCurve.reserve(static_cast<std::size_t>(
                    std::ceil((lastTime-firstTime) / 0.005)) + 2);
                for (auto time = firstTime; time < lastTime; time += 0.005)
                    renderedNote.pitchCurve.push_back({ time,
                        baseCentsAt(time)
                            + static_cast<float>(vibratoCentsAt(note, time)) });
                renderedNote.pitchCurve.push_back({ lastTime,
                    baseCentsAt(lastTime)
                        + static_cast<float>(vibratoCentsAt(note, lastTime)) });
                if(note.pitchSupportAuthored)
                {
                    for(const auto& point:note.contour)if(point.timeSeconds<0||point.timeSeconds>note.durationSeconds)
                        renderedNote.pitchCurve.push_back({point.timeSeconds,baseCentsAt(point.timeSeconds)+static_cast<float>(vibratoCentsAt(note,point.timeSeconds))});
                    for(const auto time:{0.0,note.durationSeconds})
                        renderedNote.pitchCurve.push_back({time,baseCentsAt(time)+static_cast<float>(vibratoCentsAt(note,time))});
                    std::stable_sort(renderedNote.pitchCurve.begin(),renderedNote.pitchCurve.end(),[](const auto& a,const auto& b){return a.timeSeconds<b.timeSeconds;});
                    renderedNote.pitchCurve.erase(std::unique(renderedNote.pitchCurve.begin(),renderedNote.pitchCurve.end(),[](const auto& a,const auto& b){return a.timeSeconds==b.timeSeconds;}),renderedNote.pitchCurve.end());
                }
            }
            else
            {
                renderedNote.pitchCurve.reserve(note.contour.size());
                for (const auto& point : note.contour)
                    renderedNote.pitchCurve.push_back({ point.timeSeconds,
                                                        renderedPitchCents(note, point) });
            }
        }
        const auto own = ownPitchPoints(note);
        auto first = 0.0;
        auto last = note.durationSeconds;
        if (!note.pitchControlPoints.empty() && !own.empty())
        {
            first = own.front().timeSeconds;
            last = own.front().timeSeconds;
            for (const auto& point : own)
            {
                first = std::min(first, point.timeSeconds);
                last = std::max(last, point.timeSeconds);
            }
        }
        planned.ownFirst = first;
        planned.ownLast = last;
        planned.ownPointsPlaced = !note.pitchControlPoints.empty();
        // Diagnostic isolation: the old host sampled the packet before reading
        // automatic-bridge endpoints, not through a continuous callback.
        if (nominalHostBridge) renderedNote.timelinePitchCents = {};
        return planned;
    };

    // Adjacent notes retain independent tail/head pitches, and the automatic
    // S transition carries the pitch from where one note's own line ends to
    // where the next one's begins -- the very stretch the roll draws it over.
    // With points of their own those are the points themselves, so dragging
    // them moves the transition; the automatic curve hands over an inset in
    // from each end instead, as it is drawn.  The identical absolute-pitch
    // bridge is written into both resampler requests so their overlap /
    // crossfade cannot produce two contradictory pitch trajectories.
    const auto centsAt = [](const backend::UtauNoteRenderSpec& note, double time)
    {
        if (note.timelinePitchCents) return note.timelinePitchCents(time);
        if (note.pitchCurve.empty()) return 0.0f;
        const auto right = std::lower_bound(note.pitchCurve.begin(), note.pitchCurve.end(), time,
            [](const backend::UtauPitchPoint& point, double value)
            {
                return point.timeSeconds < value;
            });
        if (right == note.pitchCurve.begin()) return right->cents;
        if (right == note.pitchCurve.end()) return note.pitchCurve.back().cents;
        const auto& left = *std::prev(right);
        const auto amount = right->timeSeconds > left.timeSeconds
            ? static_cast<float>((time - left.timeSeconds)
                / (right->timeSeconds - left.timeSeconds)) : 0.0f;
        return left.cents + (right->cents - left.cents) * amount;
    };
    const auto replaceCurveRange = [](backend::UtauNoteRenderSpec& note,
                                      double start, double end,
                                      float startMidi, float endMidi)
    {
        if (note.timelinePitchCents)
        {
            const auto existing = note.timelinePitchCents;
            note.timelinePitchCents = [existing, start, end, startMidi, endMidi,
                                      baseMidi = note.midiNote](double time)
            {
                if (time < start || time > end) return existing(time);
                const auto u = static_cast<float>(juce::jlimit(0.0, 1.0,
                    (time - start) / std::max(1.0e-6, end - start)));
                return (startMidi + (endMidi - startMidi) * u * u * (3.0f - 2.0f * u)
                        - baseMidi) * 100.0f;
            };
        }
        note.pitchCurve.erase(std::remove_if(note.pitchCurve.begin(), note.pitchCurve.end(),
            [&](const auto& point)
            {
                return point.timeSeconds >= start - 1.0e-7
                    && point.timeSeconds <= end + 1.0e-7;
            }), note.pitchCurve.end());
        const auto duration = std::max(1.0e-6, end - start);
        std::vector<backend::UtauPitchPoint> bridge;
        for (auto time = start; time < end; time += 0.005)
        {
            const auto u = static_cast<float>(juce::jlimit(0.0, 1.0,
                (time - start) / duration));
            const auto shaped = u * u * (3.0f - 2.0f * u);
            const auto midi = startMidi + (endMidi - startMidi) * shaped;
            bridge.push_back({ time, (midi - note.midiNote) * 100.0f });
        }
        bridge.push_back({ end, (endMidi - note.midiNote) * 100.0f });
        note.pitchCurve.insert(note.pitchCurve.end(), bridge.begin(), bridge.end());
        std::stable_sort(note.pitchCurve.begin(), note.pitchCurve.end(),
            [](const auto& left, const auto& right)
            {
                return left.timeSeconds < right.timeSeconds;
            });
    };
    const auto layTransition = [&](PlannedPitch& previousPlan, PlannedPitch& nextPlan)
    {
        auto& previous = previousPlan.spec;
        auto& next = nextPlan.spec;
        const auto previousEnd = previous.startSeconds + previous.durationSeconds;
        if (std::abs(previousEnd - next.startSeconds) > 0.002) return;
        // Only between two notes that both take the editor's transition: a
        // note from a UST is sung at the pitch its file gives it.
        if (!previousPlan.automaticTransition || !nextPlan.automaticTransition) return;
        // Nor between two that share a line: the line already says how one
        // turns into the other, and a bridge laid over it would sing
        // something else at the very points being dragged.
        if (previousPlan.sharesNext && nextPlan.sharesPrevious) return;
        const auto inset = automaticUtauPitchTransitionInset(
            previous.durationSeconds, next.durationSeconds);
        auto previousTailTime = previousPlan.ownPointsPlaced
            ? previousPlan.ownLast : previous.durationSeconds - inset;
        auto nextHeadTime = nextPlan.ownPointsPlaced ? nextPlan.ownFirst : inset;
        if (nominalHostBridge)
        {
            previousTailTime = previous.durationSeconds - inset;
            nextHeadTime = inset;
        }
        if (previous.durationSeconds + nextHeadTime <= previousTailTime + 1.0e-6)
        {
            // Their own lines already meet, or cross: there is no stretch
            // between them for the transition to carry, so it stays where it
            // has always been, an inset in from each side of the boundary.
            previousTailTime = previous.durationSeconds - inset;
            nextHeadTime = inset;
        }
        const auto startMidi = previous.midiNote
            + centsAt(previous, previousTailTime) / 100.0f;
        const auto endMidi = next.midiNote
            + centsAt(next, nextHeadTime) / 100.0f;
        replaceCurveRange(previous, previousTailTime,
                          previous.durationSeconds + nextHeadTime, startMidi, endMidi);
        replaceCurveRange(next, previousTailTime - previous.durationSeconds,
                          nextHeadTime, startMidi, endMidi);
    };

    // Planned over the whole track, as the roll draws them, and not over this
    // request.  A render holds only the notes selected, and pairing just
    // those left a note selected on its own with no transition into it or
    // out of it at all: sung at its first point from its very start, however
    // the line was drawn into it.  A transition only forms between touching
    // notes, so each run of touching notes that holds a note being sent is
    // planned whole, and a note is sung the same whatever else was selected
    // with it.
    std::vector<std::pair<double, const NoteData*>> timeline;   // absolute start
    for (const auto& trackClip : track.clips)
        for (const auto& note : trackClip.notes)
            timeline.emplace_back(trackClip.startSeconds + note.startSeconds, &note);
    // In the order the roll takes them in.
    std::stable_sort(timeline.begin(), timeline.end(), [](const auto& left, const auto& right)
    {
        if (std::abs(left.first - right.first) > 1.0e-9) return left.first < right.first;
        return left.first + left.second->durationSeconds
            < right.first + right.second->durationSeconds;
    });
    std::vector<PlannedPitch> planned;
    planned.reserve(timeline.size());
    for(const auto& entry:timeline)planned.push_back(planPitch(*entry.second,entry.first));
    for(std::size_t i=1;i<planned.size();++i)layTransition(planned[i-1],planned[i]);
    std::unordered_set<std::string> trackNoteIds;
    for(const auto& entry:timeline)trackNoteIds.insert(entry.second->id.toStdString());
    const auto hasIncidentPitch=std::any_of(connections.begin(),connections.end(),[&](const auto& edge)
    {return !edge.pitchCurve.empty()&&trackNoteIds.contains(edge.leftNoteId.toStdString())&&trackNoteIds.contains(edge.rightNoteId.toStdString());});
    const auto graph=hasIncidentPitch?resolvedConnectionPitchCurves(track,connections):std::vector<ResolvedConnectionPitchCurve>{};
    std::vector<AudioEngine::ProjectedNotePitch> result;result.reserve(planned.size());
    for(std::size_t i=0;i<planned.size();++i)
    {
        const auto& note=*timeline[i].second;auto renderedNote=std::move(planned[i].spec);
        AudioEngine::ProjectedNotePitch projected;projected.noteId=note.id;
        projected.absoluteStartSeconds=timeline[i].first;projected.durationSeconds=note.durationSeconds;projected.midiNote=note.midiNote;
        projected.supportStartSeconds=projected.absoluteStartSeconds;projected.supportEndSeconds=projected.absoluteStartSeconds+note.durationSeconds;
        // Rests participated in legacy context planning, never voiced output.
        if(note.nativeUnpitched||backend::isRestLyric(note.label)){result.push_back(std::move(projected));continue;}
        renderedNote.timelinePitchCents={};const auto base=renderedNote;
        const auto baseStart=base.pitchCurve.empty()?0.0:std::min(0.0,base.pitchCurve.front().timeSeconds);
        const auto baseEnd=base.pitchCurve.empty()?note.durationSeconds:std::max(note.durationSeconds,base.pitchCurve.back().timeSeconds);
        std::vector<const ResolvedConnectionPitchCurve*> incident;
        for(const auto& c:graph)if(c.active&&(c.leftNoteId==note.id||c.rightNoteId==note.id))incident.push_back(&c);
        std::stable_sort(incident.begin(),incident.end(),[](const auto* x,const auto* y){return x->fromSeconds<y->fromSeconds;});
        if(!incident.empty())
        {
            const auto graphAt=[&](const ResolvedConnectionPitchCurve& c,double local)
            {return (evaluatePitchCurve(c.points,projected.absoluteStartSeconds+local)-note.midiNote)*100.0f
                +static_cast<float>(vibratoCentsAt(note,juce::jlimit(0.0,note.durationSeconds,local)));};
            const auto inside=[&](double local){const auto t=projected.absoluteStartSeconds+local;
                return std::any_of(incident.begin(),incident.end(),[&](const auto* c){return t>=c->fromSeconds&&t<=c->toSeconds;});};
            std::vector<backend::UtauPitchPoint> samples;
            for(const auto& point:base.pitchCurve)if(!inside(point.timeSeconds))samples.push_back(point);
            for(const auto* c:incident)
            {
                const auto from=c->fromSeconds-projected.absoluteStartSeconds,to=c->toSeconds-projected.absoluteStartSeconds;
                // One-sided guards prevent graph edits leaking into the original
                // owned-domain interpolation. Extended tails hold their endpoint.
                if(from>baseStart&&from<=baseEnd){const auto t=std::nextafter(from,-std::numeric_limits<double>::infinity());samples.push_back({t,centsAt(base,t)});}
                // Keep curved graph projection below a tenth of a cent before
                // the host's integer-cent quantization. Base 5ms samples remain
                // byte-for-byte unchanged; only explicit graph windows refine.
                std::function<void(double,float,double,float,int)> appendGraph;
                appendGraph=[&](double a,float va,double b,float vb,int depth)
                {
                    bool flat=true;
                    for(const auto fraction:{.25,.5,.75})
                    {
                        const auto t=a+(b-a)*fraction;
                        if(std::abs(graphAt(*c,t)-(va+(vb-va)*static_cast<float>(fraction)))>.1f){flat=false;break;}
                    }
                    if(!flat&&depth<20&&b-a>1.e-6)
                    {const auto mid=(a+b)*.5;const auto value=graphAt(*c,mid);appendGraph(a,va,mid,value,depth+1);appendGraph(mid,value,b,vb,depth+1);}
                    else samples.push_back({b,vb});
                };
                samples.push_back({from,graphAt(*c,from)});
                std::vector<double> graphKnots{from,to};
                for(const auto& point:c->points)if(point.timeSeconds>c->fromSeconds&&point.timeSeconds<c->toSeconds)
                    graphKnots.push_back(point.timeSeconds-projected.absoluteStartSeconds);
                std::sort(graphKnots.begin(),graphKnots.end());graphKnots.erase(std::unique(graphKnots.begin(),graphKnots.end()),graphKnots.end());
                for(std::size_t k=1;k<graphKnots.size();++k)for(auto t=graphKnots[k-1];t<graphKnots[k];)
                {const auto next=std::min(graphKnots[k],t+.005);appendGraph(t,graphAt(*c,t),next,graphAt(*c,next),0);t=next;}
                for(const auto& point:c->points)if(point.timeSeconds>=c->fromSeconds&&point.timeSeconds<=c->toSeconds)
                {const auto t=point.timeSeconds-projected.absoluteStartSeconds;samples.push_back({t,graphAt(*c,t)});}
                if(to>=baseStart&&to<baseEnd){const auto t=std::nextafter(to,std::numeric_limits<double>::infinity());samples.push_back({t,centsAt(base,t)});}
            }
            if(samples.empty())samples={{baseStart,centsAt(base,baseStart)},{baseEnd,centsAt(base,baseEnd)}};
            std::stable_sort(samples.begin(),samples.end(),[](const auto& x,const auto& y){return x.timeSeconds<y.timeSeconds;});
            std::vector<backend::UtauPitchPoint> unique;
            for(const auto& point:samples)if(!unique.empty()&&unique.back().timeSeconds==point.timeSeconds)unique.back()=point;else unique.push_back(point);
            renderedNote.pitchCurve=std::move(unique);
        }
        // Both native/display readers and the packet see this bounded composition.
        const auto packet=renderedNote;
        renderedNote.timelinePitchCents=[packet,centsAt](double local){return centsAt(packet,local);};
        projected.sampledLocalCents=std::move(renderedNote.pitchCurve);projected.continuousLocalCents=std::move(renderedNote.timelinePitchCents);
        if(!projected.sampledLocalCents.empty())
        {
            projected.supportStartSeconds=projected.absoluteStartSeconds+std::min(0.0,projected.sampledLocalCents.front().timeSeconds);
            projected.supportEndSeconds=projected.absoluteStartSeconds+std::max(note.durationSeconds,projected.sampledLocalCents.back().timeSeconds);
        }
        result.push_back(std::move(projected));
    }
    return result;
}

// Pure expression adapter, separate from the bank-independent musical plan.
// Its caller retains the pre-existing DiffSinger capability decision; no other
// host receives a saved DS-specific offset merely because that field exists.
void applyDiffSingerPitchOffset(AudioEngine::ProjectedNotePitch& projection,
    const std::vector<PitchCurveEditPoint>& offsets)
{
    auto base=projection.continuousLocalCents;
    if(!base)
    {
        std::vector<PitchCurveEditPoint> cents;
        for(const auto& point:projection.sampledLocalCents)cents.push_back({point.timeSeconds,point.cents,PitchCurveShape::linear});
        base=[cents](double time){return cents.empty()?0.0f:evaluatePitchCurve(cents,time);};
    }
    const auto shifted=[base,offsets](double time){return base(time)+100.0f*evaluatePitchCurve(offsets,time);};
    auto from=-.6,to=projection.durationSeconds+.3;
    if(!projection.sampledLocalCents.empty())
    {from=std::min(from,projection.sampledLocalCents.front().timeSeconds);to=std::max(to,projection.sampledLocalCents.back().timeSeconds);}
    std::vector<double> times;
    for(auto time=from;time<to;time+=.005)times.push_back(time);
    times.push_back(to);
    for(const auto& point:offsets)times.push_back(point.timeSeconds);
    for(const auto& point:projection.sampledLocalCents)times.push_back(point.timeSeconds);
    std::sort(times.begin(),times.end());times.erase(std::unique(times.begin(),times.end()),times.end());
    projection.sampledLocalCents.clear();
    for(const auto time:times)projection.sampledLocalCents.push_back({time,shifted(time)});
    projection.continuousLocalCents=shifted;
    projection.supportStartSeconds=projection.absoluteStartSeconds+from;projection.supportEndSeconds=projection.absoluteStartSeconds+to;
}

backend::UtauRenderRequest makeUtauRequest(const ClipData& clip, const TrackData& track,
                                           const juce::File& resampler,
                                           const ProjectData& project, std::optional<bool> ownHostPitch = {},
                                           std::optional<bool> nominalHostBridge = {})
{
    backend::UtauRenderRequest request;
    request.requestedFamily = requestedVoicebankFamily(track);
    request.voicebankDirectory = track.voicebankDirectory;
    request.diffSingerCacheSession = backend::DiffSingerRenderer::cacheSession();
    request.diffSingerLanguage = track.diffSingerLanguage;
    request.diffSingerSpeaker = track.diffSingerSpeaker;
    request.diffSingerDictionary = track.diffSingerDictionary;
    request.resamplerExecutable = resampler;
    request.fourRegion = utauModeUsesRegions(track.utauMode);
    request.chineseCvvc = track.chineseCvvc && track.utauMode == UtauMode::classic && !trackIsDiffSinger(track);
    request.pianoPreviewEnabled = track.utauPianoPreview;
    request.cvvcCurveCrossfade=track.cvvcCurveCrossfade;
    request.consonantClasses = track.utauMode == UtauMode::mou;
    request.targetDurationSeconds = clip.durationSeconds;
    request.bpm = project.bpm;
    request.notes.reserve(clip.notes.size());
    auto projections=ownHostPitch.has_value()||nominalHostBridge.has_value()
        ?projectTrackPitchImpl(track,project.nativeConnections,ownHostPitch.value_or(true),nominalHostBridge.value_or(true))
        :AudioEngine::projectTrackNotePitch(track,project.nativeConnections);
    // Read pitch over the full track even when rendering a selected note or
    // one clip. Reuse the CVVC timeline adapter; musical projection stays above.
    std::unordered_map<std::string,std::shared_ptr<const backend::UtauPitchTimeline>> timelines;
    if(request.requestedFamily==backend::RequestedVoicebankFamily::utau&&!request.chineseCvvc)
    {
        std::unordered_map<std::string,const NoteData*> notes;
        for(const auto& c:track.clips)for(const auto& n:c.notes)notes.emplace(n.id.toStdString(),&n);
        std::vector<backend::UtauNoteRenderSpec> context;context.reserve(projections.size());
        for(const auto& projection:projections)
        {
            backend::UtauNoteRenderSpec n;
            const auto* source=notes.at(projection.noteId.toStdString());
            n.alias=source->nativeUnpitched?juce::String("RR"):source->label;
            n.startSeconds=projection.absoluteStartSeconds;n.durationSeconds=projection.durationSeconds;n.midiNote=projection.midiNote;
            n.pitchCurve=projection.sampledLocalCents;n.timelinePitchCents=projection.continuousLocalCents;
            context.push_back(std::move(n));
        }
        const auto runs=backend::makeUtauPitchTimelines(context);
        for(size_t i=0;i<projections.size();++i)timelines.emplace(projections[i].noteId.toStdString(),runs[i]);
    }
    std::unordered_map<std::string,AudioEngine::ProjectedNotePitch> planned;
    for(auto& projection:projections)planned.emplace(projection.noteId.toStdString(),std::move(projection));

    for (const auto& note : clip.notes)
    {
        backend::UtauNoteRenderSpec renderedNote;
        renderedNote.alias = note.nativeUnpitched ? "RR" : note.label;
        renderedNote.diffSingerTiming = note.diffSingerTiming;
        renderedNote.diffSingerPronunciation = note.diffSingerPronunciation;
        renderedNote.diffSingerContext = clip.id;
        // The engine parses flags first-wins, so the note has to come first
        // for its own settings to override the track's rather than the other
        // way round.  This is also the order UTAU itself concatenates in.
        renderedNote.flags = note.utauFlags + track.utauGlobalFlags;
        renderedNote.splice = note.utauSplice;
        renderedNote.flagCurve = note.utauFlagCurveEnabled;
        if (request.chineseCvvc)
        {
            renderedNote.cvvcBoundaries = note.cvvcBoundaries;
            renderedNote.cvvcEnvelopes = note.cvvcEnvelopes;
            for (auto& [part, curves] : renderedNote.cvvcEnvelopes)
                if (auto it = curves.find(juce::String()); it != curves.end())
                    for (auto& point : it->second)
                    {
                        const auto percent = backend::envelopeGainFromDb(point.value)
                            * juce::jlimit(0.0f, 200.0f, note.amplitudeEnvelopeBasePercent);
                        point.value = percent <= .1f ? -60.0f : 20.0f * std::log10(percent / 100.0f);
                    }
        }
        // The engine reads a curve as straight lines between the points it is
        // given, so a curved segment is sampled into enough of them to follow.
        // Its store holds 64 per curve and drops the rest silently, so the
        // budget is spent here rather than losing the tail of a long curve.
        constexpr int flagCurvePointLimit = 60;
        for (const auto& curve : note.utauFlagCurves)
        {
            if (curve.flag.startsWith("DS:")) continue;
            const auto& drawn = curve.points;
            if (drawn.empty()) {
                // Empty HIFI overrides intentionally suppress a saved legacy
                // curve while retaining it for the traditional resampler.
                if (curve.flag.startsWith("HIFI:")) renderedNote.flagCurves.emplace_back(curve.flag, std::vector<std::pair<double,double>>{});
                continue;
            }
            auto curved = 0;
            for (std::size_t index = 1; index < drawn.size(); ++index)
                if (drawn[index].shape != PitchCurveShape::linear) ++curved;
            const auto perSegment = curved > 0
                ? juce::jlimit(2, 12,
                    (flagCurvePointLimit - static_cast<int>(drawn.size())) / curved)
                : 0;
            std::vector<std::pair<double, double>> sampled;
            for (std::size_t index = 0; index < drawn.size(); ++index)
            {
                if (index > 0 && drawn[index].shape != PitchCurveShape::linear)
                    for (auto step = 1; step <= perSegment; ++step)
                    {
                        const auto at = drawn[index - 1].timeSeconds
                            + (drawn[index].timeSeconds - drawn[index - 1].timeSeconds)
                                * step / static_cast<double>(perSegment + 1);
                        sampled.emplace_back(at, flagCurveValueAt(drawn, at));
                    }
                sampled.emplace_back(drawn[index].timeSeconds, drawn[index].value);
            }
            renderedNote.flagCurves.emplace_back(curve.flag, std::move(sampled));
        }
        if (trackIsDiffSinger(track))
            renderedNote.flagCurves = sampleDiffSingerFlagCurves(note);
        renderedNote.startSeconds = note.startSeconds;
        renderedNote.durationSeconds = note.durationSeconds;
        renderedNote.midiNote = note.midiNote;
        renderedNote.gain = note.gain;
        renderedNote.nativeFormantSemitones = note.formantSemitones;
        renderedNote.nativeTension = note.tension;
        renderedNote.nativeBreath = note.breath;
        renderedNote.tailFadeMode = trackIsDiffSinger(track)?0:note.utauTailFadeMode;
        renderedNote.tailFadeSettings = note.utauTailFade;
        if(trackIsDiffSinger(track))renderedNote.tailFadeSettings.head.mode=0;
        {
            // A note with no envelope of its own still has one: a flat 100%.
            // Written out here so a base value raises it like any other.
            auto shaped = note.amplitudeEnvelope;
            const auto base = juce::jlimit(0.0f, 200.0f, note.amplitudeEnvelopeBasePercent);
            if(request.chineseCvvc && shaped.empty())renderedNote.cvvcDefaultEnvelopeGain=base/100.f;
            if (!request.chineseCvvc && shaped.empty() && std::abs(base - 100.0f) > 1.0e-6f)
                shaped = { { 0.0, 0.0f, true },
                           { std::max(0.01, note.durationSeconds), 0.0f, true } };
            shaped = scaledAmplitudeEnvelope(shaped, base);
            renderedNote.amplitudeEnvelope.reserve(shaped.size());
            for (const auto& point : shaped)
                renderedNote.amplitudeEnvelope.push_back({ point.timeSeconds, point.gainDb,
                                                          point.linearToNext });
        }
        // Fitting the envelope to the note it is now is left to the mixer,
        // which is where the note's real lead-in is known.  Carrying only the
        // closing point out to the end here stretched the fall that belongs to
        // it, so a note twice as long faded for twice as long -- a shape
        // nobody chose, and not the one the roll was drawing.
        renderedNote.consonantVelocity =
            note.utauConsonantVelocity != inheritedUtauConsonantVelocity
            ? note.utauConsonantVelocity : track.utauConsonantVelocity;
        renderedNote.preutteranceOverrideEnabled =
            note.utauPreutteranceOverrideEnabled;
        renderedNote.preutteranceSeconds = note.utauPreutteranceSeconds;
        renderedNote.overlapOverrideEnabled = note.utauOverlapOverrideEnabled;
        renderedNote.overlapSeconds = note.utauOverlapSeconds;
        renderedNote.stpSeconds = note.utauStpSeconds;
        renderedNote.modulationPercent = note.utauModulationPercent;
        renderedNote.oto = note.utauOto;
        renderedNote.jieSplitSet = note.utauJieSplitSet;
        renderedNote.jieSplit = { note.utauJieSplit1, note.utauJieSplit2,
                                  note.utauJieSplit3 };
        renderedNote.flagSplit = note.utauFlagSplit;
        renderedNote.regionFlags = { note.utauRegionFlags1, note.utauRegionFlags2,
                                     note.utauRegionFlags3, note.utauRegionFlags4 };
        renderedNote.bpm = project.tempoAtSeconds(
            clip.startSeconds + note.startSeconds);
        if(const auto found=planned.find(note.id.toStdString());found!=planned.end())
        {
            auto projection=found->second;
            if(trackIsDiffSinger(track)&&!note.diffSingerPitchOffset.empty())applyDiffSingerPitchOffset(projection,note.diffSingerPitchOffset);
            renderedNote.pitchCurve=std::move(projection.sampledLocalCents);
            renderedNote.timelinePitchCents=std::move(projection.continuousLocalCents);
        }
        if(const auto found=timelines.find(note.id.toStdString());found!=timelines.end())
            found->second->bind(renderedNote,clip.startSeconds+note.startSeconds,found->second);
        request.notes.push_back(std::move(renderedNote));
    }

    return request;
}

// A render of a selection holds only the notes selected, and starts a second
// ahead of the first of them rather than at the clip's start: enough lead-in
// for an ordinary oto.ini preutterance without a song-length buffer for a
// small marquee selection.  Each note keeps its time on the track -- the clip
// moves by what its notes do -- and the offset is returned.
double startRequestAtSelection(ClipData& requestClip, bool neural = false)
{
    if (requestClip.notes.empty()) return 0.0;
    const auto first = std::min_element(requestClip.notes.begin(), requestClip.notes.end(),
        [](const auto& left, const auto& right)
        {
            return left.startSeconds < right.startSeconds;
        });
    const auto last = std::max_element(requestClip.notes.begin(), requestClip.notes.end(),
        [](const auto& left, const auto& right)
        {
            return left.startSeconds + left.durationSeconds
                < right.startSeconds + right.durationSeconds;
        });
    const auto offset = std::max(neural ? -requestClip.startSeconds : 0.0, first->startSeconds - 1.0);
    const auto selectedEnd = last->startSeconds + last->durationSeconds + 0.25;
    requestClip.durationSeconds = std::max(0.03,
        std::min(requestClip.durationSeconds, selectedEnd) - offset);
    for (auto& note : requestClip.notes)
        note.startSeconds -= offset;
    requestClip.startSeconds += offset;
    return offset;
}
}

std::vector<AudioEngine::ProjectedNotePitch> AudioEngine::projectTrackNotePitch(
    const TrackData& track,const std::vector<NativeConnection>& connections)
{
    return projectTrackPitchImpl(track,connections);
}

AudioEngine::AudioEngine(bool initialiseOutputDevice)
{
    startupLog("AudioEngine: construct");
    // Before anyone says otherwise, the engine that travels with the
    // application is the one to use.  Without this a headless caller that
    // never sets a resampler renders through the built-in fallback and sounds
    // plausible while the packaged engine sits unused beside it.
    utauResamplerFile = bundledUtauResampler(juce::File::getSpecialLocation(
        juce::File::currentExecutableFile).getParentDirectory());
    // Read the same preferences for GUI, offline export and headless MCP.
    juce::PropertiesFile::Options options;
    options.applicationName = "HachiShifterNext";
    options.filenameSuffix = "settings";
    options.folderName = juce::SystemStats::getEnvironmentVariable("HACHI_TEST_SETTINGS_DIR", "HachiShifterNext");
    options.osxLibrarySubFolder = "Application Support";
    options.storageFormat = juce::PropertiesFile::storeAsXML;
    juce::PropertiesFile settings(options);
    setUtauResamplerFile(juce::File(settings.getValue("algorithm.utauResampler").trim().unquoted()));
    setUtauOutputDefaults(parseUtauOutputEngine(settings.getValue("algorithm.utauOutputEngine")),
        juce::File(settings.getValue("algorithm.utauWavtool").trim().unquoted()));
    hifiganModelDirectory = juce::File(settings.getValue("algorithm.hifiganPath").trim().unquoted());
    formatManager.registerBasicFormats();
    sourcePlayer.setSource(this);
    if (initialiseOutputDevice)
    {
        startupLog("AudioEngine: opening default devices");
        deviceManager.initialiseWithDefaultDevices(0, 2);
        startupLog("AudioEngine: default devices returned");
        deviceManager.addAudioCallback(&sourcePlayer);
    }
}

AudioEngine::~AudioEngine()
{
    // The device callback may be running on the driver's real-time thread.
    // Detach it before changing/destroying the AudioSourcePlayer; doing these
    // operations in the opposite order leaves a small release-build race in
    // which the driver can enter a player whose source is being torn down.
    playing.store(false, std::memory_order_release);
    deviceManager.removeAudioCallback(&sourcePlayer);
    sourcePlayer.setSource(nullptr);

    // RenderService is declared before the playback/cache members and would
    // therefore normally be destroyed after them.  Stop its jobs explicitly
    // while all callback targets and caches are still alive.
    renderService.cancelAll();
    deviceManager.closeAudioDevice();
}

bool AudioEngine::ensureOutputDevice(juce::String& error)
{
    const auto hasOutput = [this]
    {
        const auto* device = deviceManager.getCurrentAudioDevice();
        return device != nullptr
            && device->getActiveOutputChannels().countNumberOfSetBits() > 0;
    };
    if (hasOutput())
    {
        error.clear();
        return true;
    }
    // A driver may appear after startup (USB interface connected, Bluetooth
    // endpoint enabled, or Windows device service restarted).  Retry here so
    // Play does not enter a false playing state with no callback to advance
    // the transport.
    startupLog("AudioEngine: opening output device");
    error = deviceManager.initialiseWithDefaultDevices(0, 2);
    startupLog("AudioEngine: device initialisation returned: " + error);
    if (hasOutput())
    {
        error.clear();
        return true;
    }
    if (error.isEmpty()) error = "No audio output device is available";
    return false;
}

void AudioEngine::restoreDeviceState(juce::PropertiesFile& properties)
{
    const auto saved = properties.getValue("audio.deviceState");
    if (saved.isEmpty()) return;
    const auto xml = juce::parseXML(saved);
    if (xml == nullptr) return;
    startupLog("AudioEngine: restoring saved device");
    deviceManager.initialise(0, 2, xml.get(), true);
    startupLog("AudioEngine: saved device returned");
}

void AudioEngine::saveDeviceState(juce::PropertiesFile& properties) const
{
    if (const auto state = deviceManager.createStateXml())
    {
        properties.setValue("audio.deviceState", state->toString());
        properties.saveIfNeeded();
    }
}

void AudioEngine::prepareToPlay(int, double sampleRate)
{
    outputSampleRate.store(sampleRate > 0.0 ? sampleRate : 48'000.0);
}

void AudioEngine::releaseResources()
{
}

std::optional<double> AudioEngine::probeDuration(const juce::File& file)
{
    if (auto reader = std::unique_ptr<juce::AudioFormatReader>(formatManager.createReaderFor(file)))
        return static_cast<double>(reader->lengthInSamples) / reader->sampleRate;
    return std::nullopt;
}

bool AudioEngine::setAuditionFile(const juce::File& file)
{
    auto reader = std::shared_ptr<juce::AudioFormatReader>(formatManager.createReaderFor(file));
    if (reader == nullptr) return false;
    stop();
    {
        const juce::ScopedWriteLock guard(renderLock);
        auditionReader = std::move(reader);
        auditionScratch.setSize(juce::jlimit(1, 2, static_cast<int>(auditionReader->numChannels)), 2);
        // Only on the way in: swapping one audition file for another must not
        // overwrite the project position with an audition one.
        if (!auditionMode.exchange(true))
            projectTimelineSample.store(timelineSample.load());
    }
    timelineSample.store(0);
    sendChangeMessage();
    return true;
}

void AudioEngine::clearAuditionFile()
{
    stop();
    {
        const juce::ScopedWriteLock guard(renderLock);
        auditionReader.reset();
        auditionScratch.setSize(0, 0);
        auditionMode.store(false);
    }
    // Back on the project's own timeline, standing where it was left.  Zeroing
    // it here sent the playhead to the beginning every time the sample editor
    // was closed, and the next zoom then dragged the whole view after it.
    timelineSample.store(projectTimelineSample.load());
    renderService.setPlaybackPosition(position());
    sendChangeMessage();
}

bool AudioEngine::clipsJoinAt(const ClipData& clip, const ClipData& neighbour,
                              bool asNext)
{
    if (clip.notes.empty()) return false;
    // Melodyne places the two elements of a pitch join exactly back to back,
    // so a seam that is not touching is not one of them however the notes are
    // marked.  Two milliseconds is the same tolerance the fades use.
    const auto overlap = asNext
        ? clip.startSeconds + clip.durationSeconds - neighbour.startSeconds
        : neighbour.startSeconds + neighbour.durationSeconds - clip.startSeconds;
    if (std::abs(overlap) > 0.002) return false;
    return asNext ? clip.notes.back().connectedToNext
                  : clip.notes.front().connectedToPrevious;
}

namespace
{
// One bucket per millisecond of the note, holding the extremes of the samples
// inside it.  A bucket is what a waveform is drawn from, and a millisecond is
// finer than any zoom this roll offers, so the peaks survive zooming in
// without the whole rendered buffer being kept around to be re-read.
UtauNoteWaveform measureNoteWaveform(const juce::AudioBuffer<float>& buffer,
                                     double sampleRate, double startInBuffer,
                                     double durationSeconds,
                                     const std::function<float(double)>& gainAt = {},
                                     double leadInSeconds = 0.0,
                                     const std::function<float(double)>& fadesAt = {})
{
    UtauNoteWaveform waveform;
    waveform.durationSeconds = durationSeconds;
    if (sampleRate <= 0.0 || durationSeconds <= 0.0 || buffer.getNumSamples() <= 0)
        return waveform;
    const auto buckets = std::max(1, static_cast<int>(std::ceil(durationSeconds * 1000.0)));
    waveform.minima.assign(static_cast<std::size_t>(buckets), 0.0f);
    waveform.maxima.assign(static_cast<std::size_t>(buckets), 0.0f);
    waveform.unshapedMinima.assign(static_cast<std::size_t>(buckets), 0.0f);
    waveform.unshapedMaxima.assign(static_cast<std::size_t>(buckets), 0.0f);
    const auto first = static_cast<juce::int64>(std::llround(startInBuffer * sampleRate));
    const auto samples = static_cast<juce::int64>(std::llround(durationSeconds * sampleRate));
    for (int bucket = 0; bucket < buckets; ++bucket)
    {
        const auto from = first + samples * bucket / buckets;
        const auto to = first + samples * (bucket + 1) / buckets;
        auto low = 0.0f;
        auto high = 0.0f;
        auto bareLow = 0.0f;
        auto bareHigh = 0.0f;
        for (auto index = std::max<juce::int64>(0, from);
             index < std::min<juce::int64>(to, buffer.getNumSamples()); ++index)
            for (int channel = 0; channel < buffer.getNumChannels(); ++channel)
            {
                const auto sample = buffer.getSample(channel, static_cast<int>(index));
                // Read where this sample is: the piece begins a lead-in before
                // the note, and both gains are measured from the note.
                const auto localSeconds = static_cast<double>(index) / sampleRate
                    - leadInSeconds;
                // The piece before the envelope, but with the fades the mix
                // puts on it regardless: what the envelope lane's line acts
                // on, which stops where the next note takes over.
                const auto bare = fadesAt ? sample * fadesAt(localSeconds) : sample;
                bareLow = std::min(bareLow, bare);
                bareHigh = std::max(bareHigh, bare);
                // And with everything, envelope included: what is heard of it.
                const auto value = gainAt ? sample * gainAt(localSeconds) : sample;
                low = std::min(low, value);
                high = std::max(high, value);
            }
        waveform.minima[static_cast<std::size_t>(bucket)] = low;
        waveform.maxima[static_cast<std::size_t>(bucket)] = high;
        waveform.unshapedMinima[static_cast<std::size_t>(bucket)] = bareLow;
        waveform.unshapedMaxima[static_cast<std::size_t>(bucket)] = bareHigh;
    }
    return waveform;
}
}

std::uint64_t AudioEngine::utauNoteRenderHash(const NoteData& note)
{
    return noteRenderHashImpl(note);
}

std::uint64_t AudioEngine::utauNoteAudioHash(const NoteData& note)
{
    return noteRenderHashImpl(note, false);
}

std::uint64_t AudioEngine::utauNotePhonemeHash(const NoteData& note)
{
    // Envelope/FLAG edits do not change phoneme placement. Keep the editable
    // spans available while the asynchronous audio render catches up.
    NoteData timing;
    timing.label=note.label; timing.startSeconds=note.startSeconds;
    timing.durationSeconds=note.durationSeconds; timing.midiNote=note.midiNote;
    timing.cvvcBoundaries=note.cvvcBoundaries;
    timing.utauSplice=note.utauSplice;
    timing.utauConsonantVelocity=note.utauConsonantVelocity;
    timing.utauPreutteranceOverrideEnabled=note.utauPreutteranceOverrideEnabled;
    timing.utauPreutteranceSeconds=note.utauPreutteranceSeconds;
    timing.utauOverlapOverrideEnabled=note.utauOverlapOverrideEnabled;
    timing.utauOverlapSeconds=note.utauOverlapSeconds;
    timing.utauOto=note.utauOto; timing.utauStpSeconds=note.utauStpSeconds;
    return noteRenderHashImpl(timing, false);
}

std::uint64_t AudioEngine::utauPhonemeMixContextHash(const TrackData& track)
{
    auto hash = static_cast<std::uint64_t>(track.voicebankDirectory.getFullPathName().hashCode64());
    const auto add = [&](std::uint64_t value) { hash ^= value + 0x9e3779b97f4a7c15ULL + (hash<<6) + (hash>>2); };
    if (track.chineseCvvc) { add(164); add(track.cvvcCurveCrossfade); }
    add(static_cast<std::uint64_t>(track.outputEngine));
    add(static_cast<std::uint64_t>(track.utauConsonantVelocity));
    add(static_cast<std::uint64_t>(track.utauMode));
    add(static_cast<std::uint64_t>(track.chineseCvvc));
    add(static_cast<std::uint64_t>(track.outputResampler.getFullPathName().hashCode64()));
    for (const auto& clip : track.clips)
    {
        add(static_cast<std::uint64_t>(std::llround(clip.startSeconds*1000000)));
        for (const auto& note : clip.notes) add(utauNotePhonemeHash(note));
    }
    return hash;
}

std::vector<backend::UtauNoteRenderSpec> AudioEngine::utauRequestNotesForClip(
    const ProjectData& project, const juce::String& clipId)
{
    for (const auto& track : project.tracks) for (const auto& clip : track.clips)
        if (clip.id == clipId) return makeUtauRequest(clip, track, {}, project).notes;
    return {};
}

std::vector<backend::UtauNoteRenderSpec> AudioEngine::diagnosticUtauRequestNotes(
    const ProjectData& project, const juce::String& clipId,
    const std::vector<juce::String>& selection)
{
    for (const auto& track : project.tracks)
        for (const auto& clip : track.clips)
            if (clip.id == clipId)
            {
                if (selection.empty()) return makeUtauRequest(clip, track, {}, project).notes;
                if (trackIsDiffSinger(track) || (track.chineseCvvc && track.utauMode == UtauMode::classic))
                {
                    auto context=clip;
                    for (auto& note:context.notes)
                        if (std::find(selection.begin(),selection.end(),note.id)==selection.end()) note.gain=0;
                    return makeUtauRequest(context,track,{},project).notes;
                }
                auto requestClip = clip;
                std::erase_if(requestClip.notes, [&selection](const auto& note)
                {
                    return std::find(selection.begin(), selection.end(), note.id)
                        == selection.end();
                });
                startRequestAtSelection(requestClip);
                return makeUtauRequest(requestClip, track, {}, project).notes;
            }
    return {};
}

backend::RequestedVoicebankFamily AudioEngine::diagnosticUtauRequestedFamily(
    const ProjectData& project, const juce::String& clipId)
{
    for (const auto& track : project.tracks)
        for (const auto& clip : track.clips)
            if (clip.id == clipId) return makeUtauRequest(clip, track, {}, project).requestedFamily;
    return backend::RequestedVoicebankFamily::legacyAuto;
}

std::string AudioEngine::diagnosticUtauRenderKey(const ProjectData& project,
                                               const juce::String& clipId, UtauOutputEngine defaultEngine,
    const juce::File& resampler, const juce::File& wavtool)
{
    for (const auto& track : project.tracks)
        for (const auto& clip : track.clips)
            if (clip.id == clipId) return renderKey(clip, track, {}, {}, resampler, {}, defaultEngine, wavtool, nullptr, &project.nativeConnections);
    return {};
}

std::shared_ptr<const std::vector<UtauNoteWaveform>>
    AudioEngine::utauNoteWaveforms() const
{
    const juce::ScopedLock guard(utauWaveformLock);
    return utauWaveformSnapshot;
}

void AudioEngine::refreshUtauWaveformSnapshot()
{
    auto collected = std::make_shared<std::vector<UtauNoteWaveform>>();
    std::unordered_map<std::string, std::size_t> displayed;
    const auto append = [&](const std::shared_ptr<RenderedClip>& entry, bool keepParameters)
    {
        if (entry == nullptr || !entry->ready.load(std::memory_order_acquire)) return;
        const juce::ScopedLock sliceGuard(entry->sliceLock);
        for (const auto& waveform : entry->utauWaveforms)
        {
            const auto [at, inserted] = displayed.emplace(waveform.noteId.toStdString(), collected->size());
            if (inserted) collected->push_back(waveform);
            else (*collected)[at->second] = waveform;
            if (!keepParameters) (*collected)[at->second].diffSingerParameters.clear();
        }
    };
    {
        const juce::ScopedReadLock guard(renderLock);
        for (const auto& loaded : loadedClips)
        {
            // Selection is a playback scope, not the lifetime of DS analysis.
            // rebuildLoadedClips validates this history against the full score,
            // bank and render context. Read its finished results oldest first;
            // pending/failed requests cannot erase previously displayed notes.
            if (loaded->diffSinger)
                if (const auto history = diffSingerPlaybackHistory.find(loaded->clip.id.toStdString());
                    history != diffSingerPlaybackHistory.end())
                    for (const auto& completed : history->second.renders) append(completed, true);
            // The last render that was ready, which is the one being heard --
            // so an edit that has not finished rendering leaves the notes it
            // did not touch showing what they still sound like.
            const auto& entry = loaded->rendered != nullptr
                                && loaded->rendered->ready.load(std::memory_order_acquire)
                ? loaded->rendered : loaded->fallbackRendered;
            // Latest ready selection wins for the same note, without duplicates.
            // A fallback can belong to a previous bank/phrase/pitch context.
            append(entry, entry == loaded->rendered);
        }
    }
    const juce::ScopedLock guard(utauWaveformLock);
    utauWaveformSnapshot = std::move(collected);
}

std::uint64_t AudioEngine::nativeClipWaveformHash(const ClipData& viewClip, const TrackData& track)
{
    const auto clip = nativeAudioPreviewClip(viewClip);
    juce::MemoryOutputStream stream;
    writeNativePitchContext(stream, clip, track);
    stream.writeString(track.id); stream.writeString(clip.sourceFile.getFullPathName());
    stream.writeDouble(clip.startSeconds); stream.writeDouble(clip.sourceOffsetSeconds);
    stream.writeDouble(clip.sourceDurationSeconds); stream.writeDouble(clip.durationSeconds);
    stream.writeInt(static_cast<int>(track.pitchAlgorithm));
    stream.writeInt(static_cast<int>(track.stretchAlgorithm));
    stream.writeInt(static_cast<int>(track.renderOrder)); stream.writeBool(track.mergeContiguousSource); stream.writeBool(track.normalizeVolume);
    stream.writeBool(track.nsfSmoothPitchTransitions);
    stream.writeBool(track.nsfNoiseProtection);
    stream.writeInt64(static_cast<juce::int64>(clip.sourceTimeMap.size()));
    for(const auto& point:clip.sourceTimeMap)
    {stream.writeDouble(point.targetSeconds);stream.writeDouble(point.sourceSeconds);}
    stream.writeInt64(static_cast<juce::int64>(clip.notes.size()));
    for(const auto& note:clip.notes) writeNoteRenderFields(stream,note);
    std::uint64_t hash=1469598103934665603ull;
    const auto* bytes=static_cast<const unsigned char*>(stream.getData());
    for(std::size_t i=0;i<stream.getDataSize();++i){hash^=bytes[i];hash*=1099511628211ull;}
    return hash;
}

std::shared_ptr<const std::vector<NativeRenderedWaveform>> AudioEngine::nativeClipWaveforms() const
{
    const juce::ScopedLock guard(nativeWaveformLock);return nativeWaveformSnapshot;
}

void AudioEngine::refreshNativeWaveforms()
{
    const auto generation=nativeWaveformGeneration->load(std::memory_order_acquire);
    if(generation==nativeWaveformSnapshotGeneration)return;
    nativeWaveformSnapshotGeneration=generation;
    auto collected=std::make_shared<std::vector<NativeRenderedWaveform>>();
    {
        const juce::ScopedReadLock guard(renderLock);
        for(const auto& loaded:loadedClips)
        {
            // Old fallback audio may still be heard while rebuilding. It is
            // never labelled as the new render in the tuning editor.
            const auto& entry=loaded->rendered;
            if(loaded->nativeWaveformHash==0||!entry||!entry->ready.load(std::memory_order_acquire)
                ||!entry->nativePeaks)continue;
            collected->push_back({loaded->nativeWaveformHash,entry->nativePeaks});
        }
    }
    const juce::ScopedLock guard(nativeWaveformLock);nativeWaveformSnapshot=std::move(collected);
}

std::vector<std::vector<const ClipData*>> AudioEngine::glideChains(
    const std::vector<const ClipData*>& orderedClips)
{
    const auto count = orderedClips.size();
    const auto usable = [](const ClipData& clip)
    {
        return !clip.muted && !clip.notes.empty() && clip.sourceFile.existsAsFile();
    };
    std::vector<std::vector<const ClipData*>> chains;
    std::vector<bool> taken(count, false);
    for (std::size_t index = 0; index < count; ++index)
    {
        if (taken[index]) continue;
        const auto& start = *orderedClips[index];
        if (!start.glideConnectedToNext || !usable(start)) continue;
        std::vector<const ClipData*> chain { &start };
        taken[index] = true;
        const auto* cursor = &start;
        for (;;)
        {
            // The successor need not be the next clip in start order: it is
            // the one whose own preceding join says it is glide-connected and
            // whose source range continues this clip's, on the same recording.
            const ClipData* found = nullptr;
            std::size_t foundAt = 0;
            for (std::size_t other = 0; other < count; ++other)
                if (!taken[other] && usable(*orderedClips[other])
                    && orderedClips[other]->sourceFile == cursor->sourceFile
                    && orderedClips[other]->glideConnectedFromPrevious
                    && std::abs(orderedClips[other]->sourceOffsetSeconds
                        - (cursor->sourceOffsetSeconds + cursor->sourceDurationSeconds)) <= 0.01)
                {
                    found = orderedClips[other];
                    foundAt = other;
                    break;
                }
            if (found == nullptr) break;
            chain.push_back(found);
            taken[foundAt] = true;
            cursor = found;
            if (!cursor->glideConnectedToNext) break;
        }
        // One clip on its own is not a phrase, so it goes down the ordinary
        // per-clip path instead; release it so a later chain may still take it.
        if (chain.size() >= 2)
            chains.push_back(std::move(chain));
        else
            taken[index] = false;
    }
    return chains;
}

int AudioEngine::diagnosticMergedPhraseCount() const
{
    const juce::ScopedReadLock guard(renderLock);
    auto phrases = 0;
    for (const auto& [key, entry] : renderCache)
        if (key.rfind("merged|", 0) == 0) ++phrases;
    return phrases;
}

std::vector<float> AudioEngine::diagnosticNativeTargetMidi(
    const ProjectData& project, int trackIndex, int clipIndex)
{
    if (trackIndex < 0 || trackIndex >= static_cast<int>(project.tracks.size()))
        return {};
    const auto& track = project.tracks[static_cast<std::size_t>(trackIndex)];
    if (clipIndex < 0 || clipIndex >= static_cast<int>(track.clips.size()))
        return {};
    const auto& clip = track.clips[static_cast<std::size_t>(clipIndex)];
    const auto lines = sharedPitchLines(track, project.nativeConnections);
    const auto request = makeRenderRequest(clip, track, juce::File{},
                                           backend::OrtExecutionConfig{}, false, false, &lines);
    return request.targetMidi;
}
backend::Mld5FileRenderRequest AudioEngine::diagnosticNativeRequest(const ClipData& clip,const TrackData& track)
{return makeRenderRequest(clip,track,{},{});}

void AudioEngine::syncProject(const ProjectData& project, bool forExport,
                              WavExportComponent component)
{
    const juce::ScopedWriteLock guard(renderLock);
    renderComponent = component;
    componentExportError = component == WavExportComponent::full ? juce::String{}
        : componentExportIssue(project, utauResamplerFile, {},
            std::vector<juce::String>(utauRenderNoteSelection.begin(), utauRenderNoteSelection.end()),
            {}, auditionTrackId, defaultUtauOutputEngine);
    rebuildLoadedClips(project, forExport);
    auto contentDuration = 0.0;
    for (const auto& track : project.tracks)
        for (const auto& clip : track.clips)
            if (!clip.muted)
                contentDuration = std::max(contentDuration, clip.startSeconds + clip.durationSeconds);
    projectDurationSeconds.store(contentDuration);
}

bool AudioEngine::trackIsAudible(bool muted, bool solo, bool anySolo,
                                 bool referenceOnly, bool beingWorkedOn)
{
    if (muted) return false;
    if (anySolo && !solo) return false;
    // A material track is there to be worked against, not to be part of the
    // piece: it sounds while it is the one in hand and never when anything
    // else is playing.
    if (referenceOnly && !beingWorkedOn) return false;
    return true;
}

void AudioEngine::setAuditionTrack(const juce::String& trackId)
{
    const juce::ScopedWriteLock guard(renderLock);
    auditionTrackId = trackId;
}

void AudioEngine::setUtauRenderNoteSelection(const std::vector<juce::String>& noteIds)
{
    const juce::ScopedWriteLock guard(renderLock);
    renderedHistoryPlayback = false;
    // A normal marquee is an exact audition scope.  Keeping old IDs here made
    // every later marquee silently accumulate historical notes and caused the
    // rendered phrase to disagree with the visible selection.  Shift-marquee
    // is already represented by noteIds containing both old and new notes.
    ++utauRenderRequestGeneration;
    utauRenderNoteSelection.clear();
    for (const auto& id : noteIds)
        if (id.isNotEmpty())
        {
            utauRenderNoteSelection.insert(id.toStdString());
            utauRenderedNoteHistory.insert(id.toStdString());
        }
}

int AudioEngine::selectEveryUtauNote(const ProjectData& project)
{
    std::vector<juce::String> everyNote;
    for (const auto& track : project.tracks)
        if (!track.accompaniment && track.pitchAlgorithm == PitchAlgorithm::utau)
            for (const auto& clip : track.clips)
                for (const auto& note : clip.notes) everyNote.push_back(note.id);
    if (everyNote.empty()) return 0;
    setUtauRenderNoteSelection(everyNote);
    return static_cast<int>(everyNote.size());
}

bool AudioEngine::selectAllRenderedUtauNotes(const ProjectData& project)
{
    const juce::ScopedWriteLock guard(renderLock);
    renderedHistoryPlayback = true;
    ++utauRenderRequestGeneration;
    utauRenderNoteSelection = utauRenderedNoteHistory;
    // Blank UTAU notes are cheap, model-free piano previews. They must be
    // audible on the first Play after opening/importing a score, even before
    // a marquee has populated render history. Keep voiced synthesis scoped
    // to history, and never turn a DiffSinger track into a piano renderer.
    for (const auto& track : project.tracks)
        if (!track.accompaniment && track.compose && trackUsesVoicebankSynthesis(track)
            && !trackIsDiffSinger(track))
            for (const auto& clip : track.clips)
                for (const auto& note : clip.notes)
                    if (note.label.trim().isEmpty() && !note.nativeUnpitched)
                        utauRenderNoteSelection.insert(note.id.toStdString());
    return !utauRenderNoteSelection.empty() || !diffSingerPlaybackHistory.empty();
}

std::shared_ptr<AudioEngine::RenderedClip> AudioEngine::diffSingerHistoryAudio(
    const std::string& clipId, double clipStart)
{
    // Called with renderLock on the control thread. Only finished PCM enters
    // history playback; never synthesize the union of old selection IDs.
    auto& history=diffSingerPlaybackHistory[clipId];
    std::vector<std::shared_ptr<RenderedClip>> ready;
    for(const auto& state:history.renders)
        if(state->ready.load(std::memory_order_acquire)) ready.push_back(state);
    if(ready.empty())return {};
    if(ready.size()==1)return ready.front();
    const auto rate=ready.back()->sampleRate;
    double from=ready.front()->timelineOffsetSeconds,to=from;int channels=1;
    for(const auto& state:ready)
    {
        if(state->sampleRate!=rate)return ready.back();
        from=std::min(from,state->timelineOffsetSeconds);
        to=std::max(to,state->timelineOffsetSeconds+state->buffer.getNumSamples()/rate);
        channels=std::max(channels,state->buffer.getNumChannels());
    }
    auto mixed=std::make_shared<RenderedClip>();mixed->sampleRate=rate;mixed->timelineOffsetSeconds=from;
    mixed->buffer.setSize(channels,std::max(1,(int)std::ceil((to-from)*rate)));mixed->buffer.clear();
    for(const auto& state:ready)
    {
        const juce::ScopedLock guard(state->sliceLock);
        if(state->utauWaveforms.empty())continue;
        auto waves=state->utauWaveforms;
        std::sort(waves.begin(),waves.end(),[](const auto& a,const auto& b){return a.startSeconds<b.startSeconds;});
        for(std::size_t i=0;i<waves.size();++i)
        {
            const auto& wave=waves[i];
            auto first=(int)std::llround((wave.startSeconds-wave.leadInSeconds-clipStart-state->timelineOffsetSeconds)*rate);
            auto last=first+(int)std::llround(wave.durationSeconds*rate);
            if(i==0)first=std::min(first,state->firstAudibleSample);
            if(i+1==waves.size())last=std::max(last,state->lastAudibleSample+1);
            first=juce::jlimit(0,state->buffer.getNumSamples(),first);last=juce::jlimit(first,state->buffer.getNumSamples(),last);
            const auto target=(int)std::llround((state->timelineOffsetSeconds-from)*rate)+first;
            const auto count=std::min(last-first,mixed->buffer.getNumSamples()-target);
            // Later completed selections replace their owned span, rather
            // than summing the same syllable twice when selections overlap.
            if(count>0)for(int ch=0;ch<channels;++ch)
                mixed->buffer.copyFrom(ch,target,state->buffer,std::min(ch,state->buffer.getNumChannels()-1),first,count);
            std::erase_if(mixed->utauWaveforms,[&](const auto& previous){return previous.noteId==wave.noteId;});
            mixed->utauWaveforms.push_back(wave);
        }
    }
    mixed->backend=ready.back()->backend;mixed->firstAudibleSample=0;mixed->lastAudibleSample=mixed->buffer.getNumSamples()-1;
    mixed->scheduled=true;mixed->ready=true;mixed->finished=true;mixed->progress=1;
    // Keep one accumulated PCM per clip, not a whole-song buffer for every
    // marquee ever made. Pending jobs remain separate until publication.
    std::erase_if(history.renders,[](const auto& state){return state->ready.load(std::memory_order_acquire);});
    history.renders.insert(history.renders.begin(),mixed);return mixed;
}

void AudioEngine::setHifiganModelDirectory(const juce::File& directory)
{
    const juce::ScopedWriteLock guard(renderLock);
    if (hifiganModelDirectory == directory) return;
    hifiganModelDirectory = directory;
    renderCache.clear();
}

juce::File AudioEngine::bundledUtauResampler(const juce::File& executableDirectory)
{
    if (executableDirectory == juce::File{}) return {};
    // The engine this application is built to drive, in the two places a
    // portable copy would carry it.  Named outright rather than "any exe
    // here" -- the folder is full of DLLs, and guessing would be worse than
    // nothing.
    for (const auto* relative : { "engines/WCSNDM.exe", "WCSNDM.exe" })
    {
        const auto bundled = executableDirectory.getChildFile(relative);
        if (bundled.existsAsFile()) return bundled;
    }
    return {};
}

juce::File AudioEngine::resolveUtauResampler(const juce::String& configured,
                                             const juce::File& executableDirectory)
{
    // What the caller names wins, so a chosen engine is never quietly swapped.
    // Quotes and padding are tolerated because this often arrives pasted.
    const juce::File chosen(configured.trim().unquoted());
    if (chosen.existsAsFile()) return chosen;
    return bundledUtauResampler(executableDirectory);
}

void AudioEngine::setUtauOutputDefaults(UtauOutputEngine engine, const juce::File& wavtool)
{
    const juce::ScopedWriteLock guard(renderLock);
    const auto selected = engine == UtauOutputEngine::pcNsfHifigan ? engine : UtauOutputEngine::resampler;
    if (defaultUtauOutputEngine == selected && utauWavtoolFile == wavtool) return;
    defaultUtauOutputEngine = selected;
    utauWavtoolFile = wavtool;
    renderCache.clear();
}

juce::String AudioEngine::outputEngineDisplayName(const TrackData& track) const
{
    const juce::ScopedReadLock guard(renderLock);
    if (track.accompaniment || !trackUsesVoicebankSynthesis(track)) return {};
    if (trackIsDiffSinger(track)) return "DiffSinger";
    if (effectiveUtauOutputEngine(track, defaultUtauOutputEngine) == UtauOutputEngine::pcNsfHifigan)
        return juce::String::fromUTF8("HiFisampler（PC-NSF-HiFiGAN）");
    const auto engine = effectiveUtauResampler(track, utauResamplerFile);
    return engine == juce::File{} ? juce::String::fromUTF8("重采样器未配置") : engine.getFileNameWithoutExtension();
}

void AudioEngine::setUtauResamplerFile(const juce::File& executable)
{
    const juce::ScopedWriteLock guard(renderLock);
    // A settings value carried over from another machine names a file that is
    // not there; fall back rather than render through nothing.
    const auto resolved = executable.existsAsFile()
        ? executable
        : bundledUtauResampler(juce::File::getSpecialLocation(
              juce::File::currentExecutableFile).getParentDirectory());
    if (utauResamplerFile == resolved) return;
    utauResamplerFile = resolved;
    renderCache.clear();
}

void AudioEngine::setInferenceConfiguration(backend::InferenceBackend inference, int deviceIndex)
{
    const backend::OrtExecutionConfig next {
        inference, deviceIndex, std::max(1, juce::SystemStats::getNumCpus() - 1)
    };
    const juce::ScopedWriteLock guard(renderLock);
    if (inferenceConfiguration.requested == next.requested
        && inferenceConfiguration.deviceIndex == next.deviceIndex
        && inferenceConfiguration.intraOpThreads == next.intraOpThreads)
        return;
    inferenceConfiguration = next;
    renderCache.clear();
}

void AudioEngine::rebuildLoadedClips(const ProjectData& sourceProject, bool forExport)
{
    std::optional<ProjectData> contentProject;
    for (const auto& track : sourceProject.tracks)
        if (std::any_of(track.clips.begin(), track.clips.end(), [](const auto& clip)
            { return clip.audioDurationSeconds >= 0.0 || !clip.parts.empty(); })) { contentProject = sourceProject; break; }
    if (contentProject) expandProjectClipParts(*contentProject);
    if (contentProject)
        for (auto& track : contentProject->tracks)
        {
            // Voicebanks synthesize notes placed in the extended empty area.
            const auto voicebank = trackUsesVoicebankSynthesis(track);
            if (voicebank) continue;
            std::erase_if(track.clips, [](const auto& clip) { return clip.audioDurationSeconds == 0.0; });
            for (auto& clip : track.clips)
                if (clip.audioDurationSeconds >= 0.0)
                {
                    const auto offset = clip.audioStartSeconds;
                    clip.startSeconds += offset;
                    clip.durationSeconds = clip.audioDurationSeconds;
                    for (auto& note : clip.notes) note.startSeconds -= offset;
                    for (auto& point : clip.sourceTimeMap) point.targetSeconds -= offset;
                    shiftClipGainEnvelopes(clip,-offset);
                    clip.audioStartSeconds = 0.0; clip.audioDurationSeconds = -1.0;
                }
        }
    const auto& project = contentProject ? *contentProject : sourceProject;
    const auto dsOptions = backend::DiffSingerRenderer::inferenceOptions(forExport);
    std::unordered_set<std::string> projectNoteIds;
    std::unordered_set<std::string> projectClipIds;
    for (const auto& track : project.tracks)
        for (const auto& clip : track.clips)
        {
            projectClipIds.insert(clip.id.toStdString());
            for (const auto& note : clip.notes)
                projectNoteIds.insert(note.id.toStdString());
        }
    std::erase_if(utauRenderNoteSelection, [&projectNoteIds](const auto& id)
    {
        return !projectNoteIds.contains(id);
    });
    std::erase_if(utauRenderedNoteHistory, [&projectNoteIds](const auto& id)
    {
        return !projectNoteIds.contains(id);
    });
    std::erase_if(playbackFallbackByClip, [&projectClipIds](const auto& item)
    {
        return !projectClipIds.contains(item.first);
    });
    std::erase_if(diffSingerPlaybackHistory,[&projectClipIds](const auto& item){return !projectClipIds.contains(item.first);});
    for (const auto& loaded : loadedClips)
    {
        auto ready = loaded->rendered != nullptr
                && loaded->rendered->ready.load(std::memory_order_acquire)
            ? loaded->rendered : loaded->fallbackRendered;
        if (ready != nullptr && ready->ready.load(std::memory_order_acquire))
            playbackFallbackByClip[loaded->clip.id.toStdString()] = std::move(ready);
    }
    loadedClips.clear();
    trackMeters.clear();
    std::unordered_map<std::string, std::shared_ptr<juce::AudioFormatReader>> readers;
    std::unordered_set<std::string> activeRenderKeys;
    // Submit the complete snapshot before any worker picks its next task, so
    // the first track in project order cannot jump ahead of the playhead.
    renderService.setPlaybackPosition(static_cast<double>(auditionMode.load()
        ? projectTimelineSample.load() : timelineSample.load()) / outputSampleRate.load());
    renderService.beginUpdate();
    struct QueueUpdateScope
    {
        backend::RenderService& service;
        const std::unordered_set<std::string>& keys;
        ~QueueUpdateScope() { service.endUpdate(keys); }
    };
    const QueueUpdateScope queueUpdate { renderService, activeRenderKeys };
    // Hand one clip its span of a decoded phrase.  The audible range is found
    // here rather than copied from the phrase, because it is asked per clip.
    const auto sliceInto = [generation=nativeWaveformGeneration](const RenderedClip& phrase,
                              const RenderedClip::SliceTarget& target)
    {
        if (target.clip == nullptr || phrase.buffer.getNumSamples() <= 0) return;
        const auto channels = phrase.buffer.getNumChannels();
        const auto copied = std::min(target.sampleCount,
            std::max(0, phrase.buffer.getNumSamples() - target.startSample));
        target.clip->buffer.setSize(channels, std::max(1, target.sampleCount));
        target.clip->buffer.clear();
        for (int channel = 0; channel < channels; ++channel)
            target.clip->buffer.copyFrom(channel, 0, phrase.buffer, channel,
                                         target.startSample, copied);
        auto firstAudible = target.clip->buffer.getNumSamples();
        auto lastAudible = -1;
        constexpr auto audibleThreshold = 1.0e-5f;
        for (int channel = 0; channel < channels; ++channel)
            for (int sample = 0; sample < target.clip->buffer.getNumSamples(); ++sample)
                if (std::abs(target.clip->buffer.getSample(channel, sample)) > audibleThreshold)
                {
                    firstAudible = std::min(firstAudible, sample);
                    lastAudible = std::max(lastAudible, sample);
                }
        target.clip->firstAudibleSample = firstAudible;
        target.clip->lastAudibleSample = lastAudible;
        target.clip->sampleRate = phrase.sampleRate;
        target.clip->nativePeaks = measureNativeRenderedPeaks(target.clip->buffer, phrase.sampleRate);
        target.clip->backend = phrase.backend;
        target.clip->warning = phrase.warning;
        target.clip->progress.store(1.0f, std::memory_order_release);
        target.clip->ready.store(true, std::memory_order_release);
        target.clip->finished.store(true, std::memory_order_release);
        generation->fetch_add(1, std::memory_order_release);
    };
    // Tracks sharing a bank share one metadata scan in this update.
    std::unordered_map<std::string, std::string> updateBankStamps;
    std::unordered_set<std::string> changedBanks;
    const auto anySolo = std::any_of(project.tracks.begin(), project.tracks.end(),
                                     [](const auto& track) { return track.solo; });
    for (const auto& track : project.tracks)
    {
        auto meter = std::make_shared<std::atomic<float>>(0.0f);
        const auto nativePitchLines = trackShowsAllNativeRegions(track) ? sharedPitchLines(track, project.nativeConnections) : SharedPitchLines{};
        const auto nativeEnvelopes = nativeSharedEnvelopes(track);
        trackMeters[track.id.toStdString()] = meter;
        if (!trackIsAudible(track.muted, track.solo, anySolo, track.referenceOnly,
                            track.id == auditionTrackId))
            continue;
        std::vector<const ClipData*> orderedClips;
        orderedClips.reserve(track.clips.size());
        for (const auto& clip : track.clips) orderedClips.push_back(&clip);
        // Empty tracks have no render requests or OTO dependencies to stamp.
        if (orderedClips.empty()) continue;
        std::stable_sort(orderedClips.begin(), orderedClips.end(), [](const auto* left, const auto* right)
        {
            return left->startSeconds < right->startSeconds;
        });
        const auto bankPath = trackUsesVoicebankSynthesis(track)
            ? track.voicebankDirectory.getFullPathName().toStdString() : std::string{};
        auto [updateStamp, firstTrack] = updateBankStamps.try_emplace(bankPath);
        if (firstTrack) {
            updateStamp->second = voicebankRenderStamp(track);
            if (trackUsesVoicebankSynthesis(track)) {
                auto [it, inserted] = voicebankRenderStamps.try_emplace(bankPath, updateStamp->second);
                if (!inserted && it->second != updateStamp->second) {
                    it->second = updateStamp->second;
                    changedBanks.insert(bankPath);
                    backend::UtauRenderer::recheckVoicebankFiles(track.voicebankDirectory);
                }
            }
        }
        const auto& bankStamp = updateStamp->second;
        const auto bankChanged = changedBanks.contains(bankPath);
        const auto count = orderedClips.size();
        // stretchSpliceThenPitch: a chain of clips joined by Melodyne pitch
        // joins is decoded in one pass instead of being spliced afterwards.
        // Only the neural decoder is worth doing this for -- it is the one
        // whose phase and mel continuity a splice actually breaks.
        struct PendingGroup
        {
            std::vector<const ClipData*> clips;
            std::vector<std::shared_ptr<RenderedClip>> rendered;
        };
        std::vector<PendingGroup> pendingGroups;
        std::vector<bool> inGlideGroup(count, false);
        const auto mergedMode = track.compose && !track.accompaniment
            && (track.renderOrder == RenderOrder::stretchSpliceThenPitch || track.mergeContiguousSource)
            && track.pitchAlgorithm == PitchAlgorithm::nsfHifigan;
        if (mergedMode)
            for (auto& chain : glideChains(orderedClips))
            {
                PendingGroup group;
                for (const auto* member : chain)
                {
                    group.clips.push_back(member);
                    for (std::size_t index = 0; index < count; ++index)
                        if (orderedClips[index] == member) { inGlideGroup[index] = true; break; }
                }
                pendingGroups.push_back(std::move(group));
            }
        for (std::size_t clipIndex = 0; clipIndex < count; ++clipIndex)
        {
            const auto& clip = *orderedClips[clipIndex];
            // A voicebank track drives synthesis from note labels + OTO rather
            // than a placed recording.  It may synthesise through the classic
            // resampler (PitchAlgorithm::utau) or natively through the one
            // NSF-HiFiGAN renderer (nsfHifigan + a voicebank directory).  Both
            // need the same note-driven request path; only the final render
            // call differs.
            const auto classicUtau = !track.accompaniment && track.pitchAlgorithm == PitchAlgorithm::utau;
            const auto nsfVoicebank = trackUsesVoicebankSynthesis(track) && !trackIsDiffSinger(track)
                && effectiveUtauOutputEngine(track, defaultUtauOutputEngine) == UtauOutputEngine::pcNsfHifigan;
            const auto utauTrack = classicUtau || nsfVoicebank;
            if (clip.muted || (!utauTrack && !clip.sourceFile.existsAsFile())) continue;
            std::shared_ptr<juce::AudioFormatReader> reader;
            if (!utauTrack)
            {
                const auto sourceKey = clip.sourceFile.getFullPathName().toStdString();
                reader = readers[sourceKey];
                if (reader == nullptr)
                {
                    reader.reset(formatManager.createReaderFor(clip.sourceFile));
                    if (reader == nullptr) continue;
                    readers[sourceKey] = reader;
                }
            }
            auto loaded = std::make_unique<LoadedClip>();
            loaded->clip = clip;
            loaded->trackId = track.id.toStdString();
            if(track.compose && !track.accompaniment && !trackUsesVoicebankSynthesis(track))
                loaded->nativeWaveformHash = nativeClipWaveformHash(clip,track);
            loaded->smoothOverlaps = track.smoothOverlaps;
            const auto joinedStart = clipIndex > 0
                && clipsJoinAt(clip, *orderedClips[clipIndex - 1], false);
            const auto joinedEnd = clipIndex + 1 < orderedClips.size()
                && clipsJoinAt(clip, *orderedClips[clipIndex + 1], true);
            std::optional<backend::Mld5FileRenderRequest> nativeRequest;
            std::optional<std::string> nativeCacheKey;
            auto preservesSource = false;
            if (reader && trackShowsAllNativeRegions(track) && !inGlideGroup[clipIndex])
            {
                if (!clip.notes.empty())
                    nativeCacheKey = renderKey(clip, track, hifiganModelDirectory, inferenceConfiguration,
                        utauResamplerFile, dsOptions, defaultUtauOutputEngine, utauWavtoolFile, &bankStamp, &project.nativeConnections)
                        + (renderComponent == WavExportComponent::full ? std::string{}
                            : "|export-component=" + std::to_string(static_cast<int>(renderComponent)));
                const auto cached = nativeCacheKey ? renderCache.find(*nativeCacheKey) : renderCache.end();
                if (cached != renderCache.end() && cached->second)
                    preservesSource = cached->second->preservesNativeSource;
                else
                {
                    nativeRequest = makeRenderRequest(clip, track, hifiganModelDirectory,
                        inferenceConfiguration, joinedStart, joinedEnd, &nativePitchLines, &nativeEnvelopes);
                    nativeRequest->exportComponent = renderComponent;
                    const auto sourceStart = juce::jlimit<juce::int64>(0, reader->lengthInSamples - 1,
                        static_cast<juce::int64>(std::llround(clip.sourceOffsetSeconds * reader->sampleRate)));
                    const auto sourceSamples = static_cast<int>(std::max<juce::int64>(1,
                        std::min(static_cast<juce::int64>(std::llround(
                            std::max(0.001, nativeRequest->sourceDurationSeconds) * reader->sampleRate)),
                            reader->lengthInSamples - sourceStart)));
                    const auto targetSamples = std::max(1, static_cast<int>(std::llround(
                        std::max(0.001, clip.durationSeconds) * reader->sampleRate)));
                    preservesSource = backend::canPreserveNativeSource(*nativeRequest,
                        reader->sampleRate, sourceSamples, targetSamples);
                }
            }
            // A raw native recording already carries its original attack and
            // release. Only processed audio needs an implicit boundary guard;
            // explicitly authored clip fades still apply in either case.
            if (!preservesSource)
            {
                const auto compactDeclick = std::min(0.0025, loaded->clip.durationSeconds * 0.5);
                loaded->clip.fadeInSeconds = std::max(loaded->clip.fadeInSeconds, compactDeclick);
                loaded->clip.fadeOutSeconds = std::max(loaded->clip.fadeOutSeconds, compactDeclick);
            }
            // A seam inside a decoded phrase was never cut, so there is nothing
            // there to fade across; the mixer must lay these buffers down flat.
            if (inGlideGroup[clipIndex])
            {
                loaded->smoothOverlaps = false;
                loaded->clip.fadeInSeconds = 0.0;
                loaded->clip.fadeOutSeconds = 0.0;
                loaded->clip.crossfadeInSeconds = 0.0;
                loaded->clip.crossfadeOutSeconds = 0.0;
            }
            if (track.smoothOverlaps && !inGlideGroup[clipIndex] && clipIndex > 0)
            {
                const auto& previous = *orderedClips[clipIndex - 1];
                const auto overlap = previous.startSeconds + previous.durationSeconds - clip.startSeconds;
                if (overlap > 1.0e-6)
                    loaded->clip.fadeInSeconds = std::max(loaded->clip.fadeInSeconds,
                        std::min({ overlap, 0.1, loaded->clip.durationSeconds }));
                else if (joinedStart && clip.crossfadeInSeconds <= 1.0e-6)
                    loaded->clip.fadeInSeconds = std::max(loaded->clip.fadeInSeconds,
                        std::min(0.006, loaded->clip.durationSeconds * 0.5));
            }
            if (track.smoothOverlaps && !inGlideGroup[clipIndex]
                && clipIndex + 1 < orderedClips.size())
            {
                const auto& next = *orderedClips[clipIndex + 1];
                const auto overlap = clip.startSeconds + clip.durationSeconds - next.startSeconds;
                if (overlap > 1.0e-6)
                    loaded->clip.fadeOutSeconds = std::max(loaded->clip.fadeOutSeconds,
                        std::min({ overlap, 0.1, loaded->clip.durationSeconds }));
                else if (joinedEnd && clip.crossfadeOutSeconds <= 1.0e-6)
                    loaded->clip.fadeOutSeconds = std::max(loaded->clip.fadeOutSeconds,
                        std::min(0.006, loaded->clip.durationSeconds * 0.5));
            }
            // Adjacent pieces of the same unprocessed recording already have
            // continuous samples. A fresh de-click on each cut would make a dip.
            const auto continuousRawSeam = [&](const ClipData& a, const ClipData& b)
            {
                if ((!track.accompaniment && track.compose) || a.muted || b.muted
                    || a.sourceFile != b.sourceFile || std::abs(a.gain - b.gain) > 1.0e-6f
                    || a.fadeOutSeconds > 0.0 || b.fadeInSeconds > 0.0
                    || a.crossfadeOutSeconds > 0.0 || b.crossfadeInSeconds > 0.0
                    || std::abs(a.startSeconds + a.durationSeconds - b.startSeconds) > 1.0e-8) return false;
                const auto rate = [&](const ClipData& c)
                {
                    return track.accompaniment || c.sourceDurationSeconds <= 1.0e-9 ? 1.0
                        : c.sourceDurationSeconds / std::max(0.001, c.durationSeconds);
                };
                return std::abs(rate(a) - rate(b)) < 1.0e-8
                    && std::abs(a.sourceOffsetSeconds + a.durationSeconds * rate(a) - b.sourceOffsetSeconds) < 1.0e-8;
            };
            if (clipIndex > 0 && continuousRawSeam(*orderedClips[clipIndex - 1], clip))
                loaded->clip.fadeInSeconds = 0.0;
            if (clipIndex + 1 < count && continuousRawSeam(clip, *orderedClips[clipIndex + 1]))
                loaded->clip.fadeOutSeconds = 0.0;
            loaded->trackGain = track.volume;
            loaded->trackPan = juce::jlimit(-1.0f, 1.0f, track.pan);
            loaded->diffSinger = trackIsDiffSinger(track);
            loaded->accompaniment = track.accompaniment;
            loaded->meter = meter;
            loaded->reader = reader;
            if(loaded->diffSinger && !forExport && renderComponent==WavExportComponent::full)
            {
                auto& history=diffSingerPlaybackHistory[clip.id.toStdString()];
                const auto context=renderKey(clip,track,hifiganModelDirectory,inferenceConfiguration,
                    utauResamplerFile,dsOptions,defaultUtauOutputEngine,utauWavtoolFile,&bankStamp,&project.nativeConnections);
                if(history.contextKey!=context){history={};history.contextKey=context;}
                if(history.renders.size()>1)diffSingerHistoryAudio(clip.id.toStdString(),clip.startSeconds);
                if(renderedHistoryPlayback)
                {
                    loaded->rendered=diffSingerHistoryAudio(clip.id.toStdString(),clip.startSeconds);
                    loadedClips.push_back(std::move(loaded));
                    continue;
                }
            }
            if (const auto fallback = playbackFallbackByClip.find(clip.id.toStdString());
                fallback != playbackFallbackByClip.end())
                // Do not audition an earlier backend's cached output while a
                // newly selected backend is pending or unavailable.
                loaded->fallbackRendered.reset();
            auto renderClip = clip;
            const auto hasUtauSelection = utauTrack && !utauRenderNoteSelection.empty();
            // UTAU rendering is explicitly selection-driven.  Rendering every
            // MIDI note while the selection is empty can occupy the worker with
            // a whole song before a subsequently marquee-selected phrase gets
            // a chance to render.  An empty selection therefore schedules no
            // new UTAU work; previously completed audio remains available via
            // fallbackRendered/playbackFallbackByClip.
            const auto dsContext = (trackIsDiffSinger(track)
                || (track.chineseCvvc && track.utauMode == UtauMode::classic)) && hasUtauSelection
                && std::any_of(clip.notes.begin(),clip.notes.end(),[this](const auto& note)
                    {return utauRenderNoteSelection.contains(note.id.toStdString());});
            if (dsContext)
            {
                // Keep phonemizer context identical to the timing editor;
                // unselected notes condition the model but their owned audio is muted.
                for (auto& note:renderClip.notes)
                    if (!utauRenderNoteSelection.contains(note.id.toStdString())) note.gain=0;
            }
            else if (utauTrack)
                std::erase_if(renderClip.notes, [this](const auto& note)
                {
                    return !utauRenderNoteSelection.contains(note.id.toStdString());
                });
            std::stable_sort(renderClip.notes.begin(), renderClip.notes.end(),
                [](const auto& left, const auto& right)
                {
                    if (left.startSeconds != right.startSeconds)
                        return left.startSeconds < right.startSeconds;
                    // On the same beat, the shorter note leads into the longer
                    // one: a note with no length of its own is nothing but a
                    // lead-in to the note beside it.  Deciding this by id, as
                    // the last resort below does, decides it by a uuid -- so
                    // which of the two crossfaded into the other came out
                    // differently from one note to the next.
                    if (left.durationSeconds != right.durationSeconds)
                        return left.durationSeconds < right.durationSeconds;
                    if (left.midiNote != right.midiNote) return left.midiNote < right.midiNote;
                    return left.id < right.id;
                });
            // A fallback belongs to an older selection.  It is valid only for
            // a clip that also contains at least one note in the current render
            // scope; otherwise an unrelated old phrase leaks into the mix.
            //
            // A scope of nothing but rests is that same case: there is nothing
            // to sound, so the render comes back silent, a silent render never
            // becomes ready, and playback falls back to the last one that was
            // -- which is the phrase these notes used to be.  Typing RR over a
            // note that had already been played went on playing it.
            const auto anythingSounds = std::any_of(
                renderClip.notes.begin(), renderClip.notes.end(),
                [this, dsContext](const auto& note) {
                    return (!dsContext || utauRenderNoteSelection.contains(note.id.toStdString()))
                        && !backend::isRestLyric(note.label);
                });
            if (utauTrack && !anythingSounds)
                loaded->fallbackRendered.reset();
            auto requestClip = renderClip;
            const auto renderTimelineOffset = hasUtauSelection
                ? startRequestAtSelection(requestClip, trackIsDiffSinger(track) || (track.chineseCvvc && track.utauMode == UtauMode::classic)) : 0.0;
            // Every compose path must use a duration-preserving, formant-preserving render.
            // Until a selected external engine is present, the native mld5 renderer is the
            // deterministic model-free fallback rather than device-rate resampling, which
            // shifts both F0 and formants and creates the "old/child voice" failure mode.
            if (!track.accompaniment && track.compose && !renderClip.notes.empty() && inGlideGroup[clipIndex])
            {
                // Filled from the phrase once it is decoded, below.  It is not
                // a renderCache entry: the phrase is what the cache holds, and
                // this buffer is only ever a copy out of it.
                auto slice = std::make_shared<RenderedClip>();
                loaded->rendered = slice;
                for (auto& group : pendingGroups)
                    for (const auto* member : group.clips)
                        if (member == &clip)
                        {
                            group.rendered.push_back(std::move(slice));
                            break;
                        }
            }
            else if (!track.accompaniment && track.compose && !renderClip.notes.empty())
            {
                auto cacheKey = nativeCacheKey ? *nativeCacheKey : renderKey(
                    renderClip, track, hifiganModelDirectory, inferenceConfiguration,
                    utauResamplerFile, dsOptions, defaultUtauOutputEngine, utauWavtoolFile, &bankStamp, &project.nativeConnections)
                    + (renderComponent == WavExportComponent::full ? std::string{}
                        : "|export-component=" + std::to_string(static_cast<int>(renderComponent)));
                const auto hasEmptyLyrics = utauTrack && !trackIsDiffSinger(track)
                    && std::any_of(renderClip.notes.begin(), renderClip.notes.end(), [](const auto& note)
                        { return !note.nativeUnpitched && note.label.trim().isEmpty(); });
                if (hasEmptyLyrics)
                    cacheKey += forExport ? "|empty-lyrics=silent" : "|empty-lyrics=piano";
                bool forceNative = false;
                if (nsfVoicebank) {
                    forceNative = backend::HifisamplerFlags::parse(track.utauGlobalFlags).force;
                    for (const auto& note : renderClip.notes) {
                        forceNative |= backend::HifisamplerFlags::parse(note.utauFlags).force;
                        if (note.utauFlagSplit) for (const auto& text : { note.utauRegionFlags1, note.utauRegionFlags2, note.utauRegionFlags3, note.utauRegionFlags4 })
                            forceNative |= backend::HifisamplerFlags::parse(text).force;
                    }
                }
                // G bypasses the phrase cache once per explicit selection/export.
                // Rebuilding the UI for the same request must not queue it again.
                if (forceNative) cacheKey += "|G-request="+std::to_string(utauRenderRequestGeneration);
                if (bankChanged || forceNative) {
                    loaded->fallbackRendered.reset();
                    playbackFallbackByClip.erase(clip.id.toStdString());
                }
                activeRenderKeys.insert(cacheKey);
                auto& state = renderCache[cacheKey];
                if (state == nullptr) state = std::make_shared<RenderedClip>();
                state->hasEmptyLyricPreview = hasEmptyLyrics && track.utauPianoPreview && !forExport;
                if (!utauTrack) state->preservesNativeSource = preservesSource;
                state->timelineOffsetSeconds = renderTimelineOffset;
                loaded->rendered = state;
                if(loaded->diffSinger && !forExport && renderComponent==WavExportComponent::full)
                {
                    auto& history=diffSingerPlaybackHistory[clip.id.toStdString()];
                    if(std::find(history.renders.begin(),history.renders.end(),state)==history.renders.end())
                    {
                        // Drop cancelled/failed requests, retain successful scopes.
                        std::erase_if(history.renders,[](const auto& old){return !old->ready.load(std::memory_order_acquire)
                            && (!old->scheduled.load()||old->finished.load());});
                        history.renders.push_back(state);
                    }
                }
                // A failed/empty render must not remain as a permanently silent
                // cache entry.  The next selection/project sync is allowed to
                // retry it after paths or voicebank contents have been fixed.
                if (state->finished.load(std::memory_order_acquire)
                    && !state->ready.load(std::memory_order_acquire))
                {
                    state->scheduled.store(false, std::memory_order_release);
                    state->finished.store(false, std::memory_order_release);
                    state->progress.store(0.0f, std::memory_order_release);
                }
                if (!state->scheduled.exchange(true))
                {
                    backend::RenderSchedule schedule;
                    schedule.key = cacheKey;
                    schedule.startSeconds = requestClip.startSeconds;
                    schedule.endSeconds = requestClip.startSeconds + requestClip.durationSeconds;
                    if (utauTrack)
                    {
                        auto first = std::numeric_limits<double>::infinity();
                        auto last = -std::numeric_limits<double>::infinity();
                        for (const auto& note : renderClip.notes)
                            if (!dsContext || utauRenderNoteSelection.contains(note.id.toStdString()))
                            {
                                first = std::min(first, clip.startSeconds + note.startSeconds);
                                last = std::max(last, clip.startSeconds + note.startSeconds + note.durationSeconds);
                            }
                        if (std::isfinite(first))
                        { schedule.startSeconds = first; schedule.endSeconds = last; }
                    }
                    schedule.discarded = [state] { state->scheduled.store(false, std::memory_order_release); };
                    const auto publish = [state, native=!utauTrack, generation=nativeWaveformGeneration](backend::RenderedAudio result) mutable
                    {
                         if (result.buffer.getNumSamples() <= 0 || result.sampleRate <= 0.0)
                         {
                             state->warning = result.warning.isNotEmpty() ? result.warning
                                 : "Selected backend returned no audio";
                             state->backend = result.backend;
                            state->progress.store(1.0f, std::memory_order_release);
                            state->finished.store(true, std::memory_order_release);
                            return;
                        }
                        auto firstAudible = result.buffer.getNumSamples();
                        auto lastAudible = -1;
                        constexpr auto audibleThreshold = 1.0e-5f;
                        for (int channel = 0; channel < result.buffer.getNumChannels(); ++channel)
                            for (int sample = 0; sample < result.buffer.getNumSamples(); ++sample)
                                if (std::abs(result.buffer.getSample(channel, sample))
                                    > audibleThreshold)
                                {
                                    firstAudible = std::min(firstAudible, sample);
                                    lastAudible = std::max(lastAudible, sample);
                                }
                        if (lastAudible < firstAudible && !result.backend.contains("+breath")
                            && !result.backend.contains("+non-breath"))
                        {
                            state->progress.store(1.0f, std::memory_order_release);
                            state->finished.store(true, std::memory_order_release);
                            return;
                        }
                        state->buffer = std::move(result.buffer);
                        state->sampleRate = result.sampleRate;
                        state->firstAudibleSample = firstAudible;
                        state->lastAudibleSample = lastAudible;
                        state->backend = std::move(result.backend);
                        state->warning = std::move(result.warning);
                        if(native) state->nativePeaks = measureNativeRenderedPeaks(state->buffer,state->sampleRate);
                        state->ready.store(true, std::memory_order_release);
                        state->progress.store(1.0f, std::memory_order_release);
                        state->finished.store(true, std::memory_order_release);
                        if(native) generation->fetch_add(1,std::memory_order_release);
                    };
                    if (utauTrack)
                    {
                        // What each note will occupy in the buffer that comes
                        // back, and what the note looked like when it was sent.
                        struct PendingNote
                        {
                            std::size_t requestIndex;
                            juce::String id;
                            std::uint64_t hash;
                            std::uint64_t audio;
                            double startInBuffer;
                            double durationSeconds;
                            double timelineStart;
                            std::uint64_t layout;
                        };
                        auto pending = std::make_shared<std::vector<PendingNote>>();
                        // The hash has to be of the note as the project holds
                        // it, which is renderClip's copy.  requestClip is the
                        // same notes with their starts shifted back to the
                        // beginning of the trimmed buffer, and hashing those
                        // would never match what the roll asks about -- so
                        // nothing would ever be drawn for a selection that
                        // begins more than a second into the song.
                        for (std::size_t index = 0; index < requestClip.notes.size()
                                                     && index < renderClip.notes.size(); ++index)
                        {
                            const auto& sent = requestClip.notes[index];
                            const auto& asHeld = renderClip.notes[index];
                            if (backend::isRestLyric(sent.label)
                                || (dsContext && !utauRenderNoteSelection.contains(asHeld.id.toStdString()))) continue;
                            pending->push_back({ index, asHeld.id, noteRenderHash(asHeld),
                                                 utauNoteAudioHash(asHeld),
                                                 sent.startSeconds, sent.durationSeconds,
                                                 clip.startSeconds + asHeld.startSeconds, utauNotePhonemeHash(asHeld) });
                        }
                        // Each note's own audio is measured as the renderer
                        // hands it over, one at a time and before any of it is
                        // mixed.  Measuring the finished mix instead read a
                        // stretch that holds two notes wherever they overlap,
                        // and put a note's consonant -- which sounds ahead of
                        // the beat -- in the row of the note before it.
                        auto pieces = std::make_shared<std::vector<UtauNoteWaveform>>();
                        auto pieceLock = std::make_shared<juce::CriticalSection>();
                        const auto measure = [state, pieces, pieceLock, engine = this]
                            (backend::RenderedAudio result)
                        {
                            juce::ignoreUnused(result);
                            std::vector<UtauNoteWaveform> measured;
                            {
                                const juce::ScopedLock pieceGuard(*pieceLock);
                                measured = *pieces;
                            }
                            {
                                const juce::ScopedLock sliceGuard(state->sliceLock);
                                state->utauWaveforms = std::move(measured);
                            }
                            engine->utauWaveformGeneration.fetch_add(
                                1, std::memory_order_release);
                        };
                        auto request = makeUtauRequest(requestClip, track,
                            effectiveUtauResampler(track, utauResamplerFile), project,
                            diagnosticUtauOwnPitch, diagnosticUtauNominalBridge);
                        request.wavtoolExecutable = effectiveUtauWavtool(track, utauWavtoolFile);
                        request.silenceEmptyLyrics = forExport;
                        request.requireExternalResampler = track.outputResampler != juce::File{};
                        request.timelineStartSeconds = requestClip.startSeconds;
                        request.diffSingerInference = dsOptions;
                        request.exportComponent = renderComponent;
                        auto phonemes=std::make_shared<std::map<std::size_t,std::vector<backend::UtauPhonemeSpan>>>();
                        request.notePhonemes=[phonemes](std::size_t i,const auto& spans){(*phonemes)[i]=spans;};
                        auto parameters=std::make_shared<std::map<std::size_t,std::vector<FlagCurve>>>();
                        request.noteParameters=[parameters](std::size_t i,const juce::var& values) {
                            for (const auto& kind:diffSingerParameterKinds()) {
                                const auto code=juce::String(kind.flag).substring(7);
                                if (const auto* points=values[juce::Identifier(code)].getArray()) {
                                    FlagCurve curve{"DS:AUTO:"+code,{}};
                                    for (const auto& p:*points) if(p.isArray() && p.size()==2)
                                        curve.points.push_back({(double)p[0],(float)p[1]});
                                    if (!curve.points.empty()) (*parameters)[i].push_back(std::move(curve));
                                }
                            }
                        };
                        request.notePiece = [pending, pieces, pieceLock, phonemes, parameters,
                            mixContext = utauPhonemeMixContextHash(track)]
                            (std::size_t index, const juce::AudioBuffer<float>& piece,
                             double sampleRate, double leadInSeconds,
                             const std::function<float(double)>& gainAt,
                             const std::function<float(double)>& fadesAt)
                        {
                            for (const auto& note : *pending)
                            {
                                if (note.requestIndex != index) continue;
                                const auto seconds = sampleRate > 0.0
                                    ? piece.getNumSamples() / sampleRate : 0.0;
                                auto waveform = measureNoteWaveform(piece, sampleRate,
                                                                    0.0, seconds,
                                                                    gainAt, leadInSeconds,
                                                                    fadesAt);
                                waveform.noteId = note.id;
                                waveform.renderHash = note.hash;
                                waveform.audioHash = note.audio;
                                waveform.phonemeLayoutHash = note.layout;
                                waveform.phonemeMixContextHash = mixContext;
                                waveform.startSeconds = note.timelineStart;
                                waveform.leadInSeconds = leadInSeconds;
                                if (const auto found=phonemes->find(index);found!=phonemes->end()) waveform.phonemes=found->second;
                                if (const auto found=parameters->find(index);found!=parameters->end()) waveform.diffSingerParameters=found->second;
                                const juce::ScopedLock pieceGuard(*pieceLock);
                                pieces->push_back(std::move(waveform));
                                return;
                            }
                        };
                        std::weak_ptr<RenderedClip> weakState(state);
                        request.progress = [weakState](double value)
                        {
                            if (const auto current = weakState.lock())
                                current->progress.store(static_cast<float>(
                                    juce::jlimit(0.0, 1.0, value)),
                                    std::memory_order_release);
                        };
                        // publish is declared const, and a copy captured from a
                        // const variable stays const however mutable this is.
                        auto forwardMeasured = [forward = publish, measure]
                            (backend::RenderedAudio result) mutable
                        {
                            // Measured before publishing, so a roll that sees
                            // the audio become ready finds the peaks already
                            // there.
                            if (result.buffer.getNumSamples() > 0
                                && result.sampleRate > 0.0)
                                measure(result);
                            forward(std::move(result));
                        };
                        if (!diagnosticUtauTimelinePitch)
                            for (auto& note : request.notes) note.timelinePitchCents = {};
                        if (nsfVoicebank) {
                            scheduledNsfUtauRenders.fetch_add(1, std::memory_order_relaxed);
                            // A voicebank track on NSF-HiFiGAN synthesises
                            // natively through the one NSF-HiFiGAN renderer,
                            // not the classic resampler.
                            renderService.renderNsfUtau(std::move(request),
                                hifiganModelDirectory, inferenceConfiguration,
                                std::move(forwardMeasured), std::move(schedule));
                        } else
                            renderService.renderUtau(std::move(request),
                                std::move(forwardMeasured), std::move(schedule));
                    }
                    else
                    {
                        auto request = nativeRequest ? std::move(*nativeRequest)
                            : makeRenderRequest(clip, track, hifiganModelDirectory,
                                inferenceConfiguration, joinedStart, joinedEnd, &nativePitchLines, &nativeEnvelopes);
                        request.exportComponent = renderComponent;
                        request.nsfIncrementalRender = diagnosticNsfIncrementalRender.value_or(request.nsfIncrementalRender);
                        renderService.renderMld5File(std::move(request), publish, std::move(schedule));
                    }
                }
            }
            // Playback only needs clip timing/gain after the render request is
            // created.  Drop duplicated contours here so large MPD projects do
            // not keep a second full copy of every analysis point per clip.
            loaded->clip.notes.clear();
            loaded->clip.notes.shrink_to_fit();
            loadedClips.push_back(std::move(loaded));
        }
        for (auto& group : pendingGroups)
        {
            // A clip may have been dropped between grouping and loading (an
            // unreadable source), which would leave the slices misaligned.
            if (group.clips.size() < 2 || group.rendered.size() != group.clips.size()) continue;
            const auto mergedKey = mergedRenderKey(group.clips, track,
                hifiganModelDirectory, inferenceConfiguration, &project.nativeConnections);
            activeRenderKeys.insert(mergedKey);
            auto& phrase = renderCache[mergedKey];
            if (phrase == nullptr) phrase = std::make_shared<RenderedClip>();
            if (phrase->finished.load(std::memory_order_acquire)
                && !phrase->ready.load(std::memory_order_acquire))
            {
                phrase->scheduled.store(false, std::memory_order_release);
                phrase->finished.store(false, std::memory_order_release);
            }
            const auto sourceKey = group.clips.front()->sourceFile.getFullPathName().toStdString();
            const auto readerIt = readers.find(sourceKey);
            const auto fileRate = readerIt != readers.end() && readerIt->second != nullptr
                ? readerIt->second->sampleRate : outputSampleRate.load();
            std::vector<RenderedClip::SliceTarget> targets;
            targets.reserve(group.clips.size());
            auto targetOffset = 0.0;
            for (std::size_t index = 0; index < group.clips.size(); ++index)
            {
                const auto start = static_cast<int>(std::llround(targetOffset * fileRate));
                targetOffset += group.clips[index]->durationSeconds;
                const auto end = static_cast<int>(std::llround(targetOffset * fileRate));
                targets.push_back({ group.rendered[index], start, std::max(1, end - start) });
            }
            {
                const juce::ScopedLock sliceGuard(phrase->sliceLock);
                if (phrase->ready.load(std::memory_order_acquire))
                    for (const auto& target : targets)
                        sliceInto(*phrase, target);
                else
                    for (auto& target : targets)
                        phrase->pendingSlices.push_back(std::move(target));
            }
            if (!phrase->scheduled.exchange(true))
            {
                auto request = mergedRequestFor(group.clips, track, hifiganModelDirectory,
                    inferenceConfiguration, diagnosticLegacyUvCarrier, diagnosticSkipSharedPitch.value_or(false), &nativePitchLines);
                request.nsfIncrementalRender = diagnosticNsfIncrementalRender.value_or(request.nsfIncrementalRender);
                renderService.renderMld5File(std::move(request),
                    [phrase, sliceInto](backend::RenderedAudio result) mutable
                    {
                        if (result.buffer.getNumSamples() <= 0 || result.sampleRate <= 0.0)
                        {
                            const juce::ScopedLock sliceGuard(phrase->sliceLock);
                            phrase->warning = result.warning.isNotEmpty() ? result.warning
                                : "NSF merged render returned no audio";
                            for (const auto& target : phrase->pendingSlices)
                            {
                                target.clip->warning = phrase->warning;
                                target.clip->progress.store(1.0f, std::memory_order_release);
                                target.clip->finished.store(true, std::memory_order_release);
                            }
                            phrase->pendingSlices.clear();
                            phrase->finished.store(true, std::memory_order_release);
                            return;
                        }
                        phrase->buffer = std::move(result.buffer);
                        phrase->sampleRate = result.sampleRate;
                        phrase->backend = std::move(result.backend);
                        phrase->warning = std::move(result.warning);
                        const juce::ScopedLock sliceGuard(phrase->sliceLock);
                        phrase->ready.store(true, std::memory_order_release);
                        phrase->finished.store(true, std::memory_order_release);
                        for (const auto& target : phrase->pendingSlices)
                            sliceInto(*phrase, target);
                        phrase->pendingSlices.clear();
                    }, { mergedKey, group.clips.front()->startSeconds,
                         group.clips.back()->startSeconds + group.clips.back()->durationSeconds,
                         [phrase] { phrase->scheduled.store(false, std::memory_order_release); } });
            }
        }
    }
    std::erase_if(renderCache, [&](const auto& item)
    {
        return renderComponent == WavExportComponent::full && !activeRenderKeys.contains(item.first);
    });
    // Which entries are current has changed, so which notes have peaks has too.
    nativeWaveformGeneration->fetch_add(1, std::memory_order_release);
    utauWaveformGeneration.fetch_add(1, std::memory_order_release);
}

void AudioEngine::refreshUtauWaveforms()
{
    // Nothing has landed and no clip has moved: the snapshot in hand is still
    // the answer, and rebuilding it would copy every note's peaks for nothing.
    if (utauWaveformGeneration.load(std::memory_order_acquire)
        == utauWaveformSnapshotGeneration)
        return;
    utauWaveformSnapshotGeneration = utauWaveformGeneration.load(std::memory_order_acquire);
    refreshUtauWaveformSnapshot();
}

float AudioEngine::fadeEnvelope(const ClipData& clip, double localSeconds)
{
    auto gain = 1.0f;
    // Melodyne successive-join amplitude transitions are LINEAR complementary
    // fades (element amplitudeFadeIn/OutShapePow are always 1.0).  Each joined
    // element is back-to-back with its partner, so the fade-in of the following
    // element and the fade-out of the preceding element meet at the boundary
    // and together span the full MUSuccessiveJoin.amplitudeTransitionDuration.
    if (clip.crossfadeInSeconds > 1.0e-6)
    {
        const auto phase = static_cast<float>(juce::jlimit(0.0, 1.0,
            localSeconds / clip.crossfadeInSeconds));
        gain *= phase;
    }
    if (clip.crossfadeOutSeconds > 1.0e-6)
    {
        const auto phase = static_cast<float>(juce::jlimit(0.0, 1.0,
            (clip.durationSeconds - localSeconds) / clip.crossfadeOutSeconds));
        gain *= phase;
    }
    if (clip.fadeInSeconds > 1.0e-6)
    {
        const auto phase = static_cast<float>(juce::jlimit(0.0, 1.0,
            localSeconds / clip.fadeInSeconds));
        gain *= phase * phase * (3.0f - 2.0f * phase);
    }
    if (clip.fadeOutSeconds > 1.0e-6)
    {
        const auto phase = static_cast<float>(juce::jlimit(0.0, 1.0,
            (clip.durationSeconds - localSeconds) / clip.fadeOutSeconds));
        gain *= phase * phase * (3.0f - 2.0f * phase);
    }
    return gain;
}

void AudioEngine::getNextAudioBlock(const juce::AudioSourceChannelInfo& info)
{
    info.clearActiveBufferRegion();
    if (!playing.load() || info.buffer == nullptr || info.numSamples <= 0) return;

    const auto sampleRate = outputSampleRate.load();
    const auto blockStartSample = timelineSample.load();
    const auto blockStart = static_cast<double>(blockStartSample) / sampleRate;
    const auto blockEnd = static_cast<double>(blockStartSample + info.numSamples) / sampleRate;
    if (!auditionMode.load(std::memory_order_relaxed) && !offlineRendering.load(std::memory_order_relaxed))
        renderService.setPlaybackPosition(blockStart);
    const juce::ScopedReadLock guard(renderLock);
    // Export-only renders must never become the ordinary transport's voice.
    // The offline writer enables offlineRendering before requesting blocks.
    if (renderComponent != WavExportComponent::full && !offlineRendering.load()
        && !auditionMode.load()) return;

    for (const auto& [_, meter] : trackMeters)
        meter->store(meter->load(std::memory_order_relaxed) * 0.88f, std::memory_order_relaxed);

    if (auditionMode.load() && auditionReader != nullptr)
    {
        const auto readerRate = auditionReader->sampleRate;
        const auto firstSourcePosition = blockStart * readerRate;
        if (firstSourcePosition >= static_cast<double>(auditionReader->lengthInSamples))
        {
            playing.store(false);
            return;
        }
        const auto availableSeconds = (static_cast<double>(auditionReader->lengthInSamples)
                                       - firstSourcePosition) / readerRate;
        const auto outputCount = juce::jlimit(0, info.numSamples,
            static_cast<int>(std::ceil(availableSeconds * sampleRate)));
        const auto lastSourcePosition = firstSourcePosition
            + static_cast<double>(std::max(0, outputCount - 1)) * readerRate / sampleRate;
        const auto sourceBase = static_cast<juce::int64>(std::floor(firstSourcePosition));
        const auto sourceCount = std::max(2, static_cast<int>(std::ceil(lastSourcePosition))
                                            - static_cast<int>(sourceBase) + 2);
        const auto sourceChannels = juce::jlimit(1, 2, static_cast<int>(auditionReader->numChannels));
        auditionScratch.setSize(sourceChannels, sourceCount, false, false, true);
        auditionScratch.clear();
        auditionReader->read(&auditionScratch, 0, sourceCount, sourceBase, true, sourceChannels > 1);
        for (int outputOffset = 0; outputOffset < outputCount; ++outputOffset)
        {
            const auto sourcePosition = firstSourcePosition
                + static_cast<double>(outputOffset) * readerRate / sampleRate - static_cast<double>(sourceBase);
            const auto leftIndex = juce::jlimit(0, sourceCount - 1, static_cast<int>(std::floor(sourcePosition)));
            const auto rightIndex = juce::jmin(sourceCount - 1, leftIndex + 1);
            const auto fraction = static_cast<float>(sourcePosition - std::floor(sourcePosition));
            const auto interpolate = [&, leftIndex, rightIndex, fraction](int channel)
            {
                const auto* samples = auditionScratch.getReadPointer(channel);
                return samples[leftIndex] + (samples[rightIndex] - samples[leftIndex]) * fraction;
            };
            const auto left = interpolate(0);
            const auto right = sourceChannels > 1 ? interpolate(1) : left;
            info.buffer->setSample(0, info.startSample + outputOffset, left);
            if (info.buffer->getNumChannels() > 1)
                info.buffer->setSample(1, info.startSample + outputOffset, right);
        }
        timelineSample.fetch_add(outputCount);
        if (outputCount < info.numSamples) playing.store(false);
        if (!offlineRendering.load(std::memory_order_relaxed)) sendChangeMessage();
        return;
    }

    // A not-yet-rendered voice must not let the accompaniment run ahead. Keep
    // the common transport stationary, without waiting on this audio thread.
    if (!offlineRendering.load(std::memory_order_relaxed)
        && pendingAudioInRange(blockStart, blockEnd)) return;

    std::unordered_map<std::string, std::vector<float>> overlapEnvelopeSums;
    std::unordered_map<std::string, std::vector<unsigned short>> overlapCounts;
    // Both ends use the same half-open sample boundary. floor(start) and
    // ceil(end) otherwise count a shared seam twice, even at an exact sample
    // time when subtraction from blockStart introduces rounding error.
    const auto blockOffsetAt = [&](double seconds)
    {
        return static_cast<int>(juce::jlimit(0.0, static_cast<double>(info.numSamples),
            std::ceil(seconds * sampleRate - 1.0e-7) - static_cast<double>(blockStartSample)));
    };
    for (const auto& loaded : loadedClips)
    {
        if (!exportTrackFilter.empty() && loaded->trackId != exportTrackFilter) continue;
        if (!loaded->smoothOverlaps) continue;
        const auto& clip = loaded->clip;
        const auto clipEnd = clip.startSeconds + clip.durationSeconds;
        const auto overlapStart = std::max(blockStart, clip.startSeconds);
        const auto overlapEnd = std::min(blockEnd, clipEnd);
        if (overlapEnd <= overlapStart) continue;
        auto& sums = overlapEnvelopeSums[loaded->trackId];
        auto& counts = overlapCounts[loaded->trackId];
        if (sums.empty()) sums.assign(static_cast<std::size_t>(info.numSamples), 0.0f);
        if (counts.empty()) counts.assign(static_cast<std::size_t>(info.numSamples), 0);
        const auto begin = blockOffsetAt(overlapStart);
        const auto end = blockOffsetAt(overlapEnd);
        for (auto output = begin; output < end; ++output)
        {
            const auto absoluteSeconds = blockStart + static_cast<double>(output) / sampleRate;
            sums[static_cast<std::size_t>(output)] += fadeEnvelope(
                clip, absoluteSeconds - clip.startSeconds);
            ++counts[static_cast<std::size_t>(output)];
        }
    }

    const auto smoothedGain = [&](const LoadedClip& loaded, double localSeconds,
                                  int blockOffset)
    {
        auto envelope = fadeEnvelope(loaded.clip, localSeconds);
        if (loaded.smoothOverlaps)
        {
            const auto sums = overlapEnvelopeSums.find(loaded.trackId);
            const auto counts = overlapCounts.find(loaded.trackId);
            if (sums != overlapEnvelopeSums.end() && counts != overlapCounts.end()
                && counts->second[static_cast<std::size_t>(blockOffset)] > 1)
            {
                const auto total = sums->second[static_cast<std::size_t>(blockOffset)];
                if (total > 1.0e-6f) envelope /= std::max(0.5f, total);
            }
        }
        return loaded.clip.gain * envelope
            * clipGainEnvelopeAt(loaded.clip,localSeconds);
    };

    for (auto& loaded : loadedClips)
    {
        // One file per track: everything else is passed over rather than
        // silenced, so the overlap sums above stay this track's own.
        if (!exportTrackFilter.empty() && loaded->trackId != exportTrackFilter) continue;
        const auto& clip = loaded->clip;
        auto renderedState = loaded->rendered != nullptr
                && loaded->rendered->ready.load(std::memory_order_acquire)
            ? loaded->rendered : loaded->fallbackRendered;
        const auto clipEnd = clip.startSeconds + clip.durationSeconds;
        const auto soundStart = loaded->diffSinger && renderedState != nullptr
            ? std::min(clip.startSeconds,clip.startSeconds+renderedState->timelineOffsetSeconds) : clip.startSeconds;
        const auto overlapStart = std::max(blockStart, soundStart);
        const auto overlapEnd = std::min(blockEnd, clipEnd);
        if (overlapEnd <= overlapStart) continue;

        const auto outputBegin = blockOffsetAt(overlapStart);
        const auto outputEnd = blockOffsetAt(overlapEnd);
        const auto outputCount = outputEnd - outputBegin;
        if (outputCount <= 0) continue;

        if (renderedState != nullptr
            && renderedState->ready.load(std::memory_order_acquire)
            && renderedState->buffer.getNumSamples() > 0)
        {
            const auto& rendered = renderedState->buffer;
            const auto renderedRate = renderedState->sampleRate;
            const auto renderedChannels = juce::jlimit(1, 2, rendered.getNumChannels());
            const auto sourceOrigin = (clip.startSeconds + renderedState->timelineOffsetSeconds) * sampleRate;
            const auto sourceIsOnGrid = std::abs(renderedRate - sampleRate) <= 1.0e-9
                && std::isfinite(sourceOrigin) && std::abs(sourceOrigin - std::round(sourceOrigin)) <= 1.0e-7;
            // Same-rate clips placed on the sample grid must read exact PCM.
            // Subtracting large seconds first otherwise creates tiny fractional
            // interpolation at exact zeros in repeated placements.
            const auto firstPosition = sourceIsOnGrid
                ? static_cast<double>(blockStartSample + outputBegin) - std::round(sourceOrigin)
                : (blockStart + static_cast<double>(outputBegin) / sampleRate
                    - clip.startSeconds - renderedState->timelineOffsetSeconds) * renderedRate;
            const auto [leftPan, rightPan] = panGains(loaded->trackPan, renderedChannels == 1);
            for (int outputOffset = 0; outputOffset < outputCount; ++outputOffset)
            {
                const auto position = firstPosition + (sourceIsOnGrid
                    ? static_cast<double>(outputOffset)
                    : static_cast<double>(outputOffset) * renderedRate / sampleRate);
                if (position < 0.0 || position >= static_cast<double>(rendered.getNumSamples()))
                    continue;
                const auto leftIndex = juce::jlimit(0, rendered.getNumSamples() - 1,
                                                     static_cast<int>(std::floor(position)));
                const auto rightIndex = std::min(rendered.getNumSamples() - 1, leftIndex + 1);
                const auto fraction = static_cast<float>(position - std::floor(position));
                const auto interpolate = [&](int channel)
                {
                    const auto* samples = rendered.getReadPointer(channel);
                    if (fraction == 0.0f) return samples[leftIndex];
                    return samples[leftIndex] + (samples[rightIndex] - samples[leftIndex]) * fraction;
                };
                const auto sourceLeft = interpolate(0);
                const auto sourceRight = renderedChannels > 1 ? interpolate(1) : sourceLeft;
                const auto absoluteSeconds = blockStart
                    + static_cast<double>(outputBegin + outputOffset) / sampleRate;
                const auto gain = smoothedGain(*loaded,
                    absoluteSeconds - clip.startSeconds, outputBegin + outputOffset)
                    * loaded->trackGain;
                const auto destination = info.startSample + outputBegin + outputOffset;
                const auto renderedLeft = sourceLeft * gain * leftPan;
                const auto renderedRight = sourceRight * gain * rightPan;
                info.buffer->addSample(0, destination, renderedLeft);
                if (info.buffer->getNumChannels() > 1)
                    info.buffer->addSample(1, destination, renderedRight);
                if (loaded->meter != nullptr)
                {
                    const auto peak = std::max(std::abs(renderedLeft), std::abs(renderedRight));
                    loaded->meter->store(std::max(loaded->meter->load(std::memory_order_relaxed), peak),
                                         std::memory_order_relaxed);
                }
            }
            continue;
        }

        // MIDI-backed UTAU clips have no AudioFormatReader.  Until their
        // asynchronous phrase render is ready they are intentionally silent.
        if (loaded->reader == nullptr) continue;

        const auto readerRate = loaded->reader->sampleRate;
        const auto sourceDuration = clip.sourceDurationSeconds > 1.0e-9
            ? clip.sourceDurationSeconds : clip.durationSeconds;
        const auto playbackRate = loaded->accompaniment ? 1.0
            : sourceDuration / std::max(0.001, clip.durationSeconds);
        const auto firstSourcePosition = clip.sourceOffsetSeconds * readerRate
            + (blockStart + static_cast<double>(outputBegin) / sampleRate - clip.startSeconds)
                * playbackRate * readerRate;
        const auto lastSourcePosition = firstSourcePosition
            + static_cast<double>(outputCount - 1) * playbackRate * readerRate / sampleRate;
        const auto sourceBase = static_cast<juce::int64>(std::floor(firstSourcePosition));
        const auto sourceCount = static_cast<int>(std::ceil(lastSourcePosition))
            - static_cast<int>(sourceBase) + 2;
        if (sourceBase < 0 || sourceCount <= 1) continue;

        const auto sourceChannels = juce::jlimit(1, 2, static_cast<int>(loaded->reader->numChannels));
        loaded->scratch.setSize(sourceChannels, sourceCount, false, false, true);
        loaded->scratch.clear();
        loaded->reader->read(&loaded->scratch, 0, sourceCount, sourceBase, true, sourceChannels > 1);

        const auto [leftPan, rightPan] = panGains(loaded->trackPan, sourceChannels == 1);
        for (int outputOffset = 0; outputOffset < outputCount; ++outputOffset)
        {
            const auto sourcePosition = firstSourcePosition
                + static_cast<double>(outputOffset) * playbackRate * readerRate / sampleRate
                - static_cast<double>(sourceBase);
            const auto leftIndex = juce::jlimit(0, sourceCount - 1, static_cast<int>(std::floor(sourcePosition)));
            const auto rightIndex = juce::jmin(sourceCount - 1, leftIndex + 1);
            const auto fraction = static_cast<float>(sourcePosition - std::floor(sourcePosition));
            const auto interpolate = [&, leftIndex, rightIndex, fraction](int channel)
            {
                const auto* samples = loaded->scratch.getReadPointer(channel);
                return samples[leftIndex] + (samples[rightIndex] - samples[leftIndex]) * fraction;
            };
            const auto sourceLeft = interpolate(0);
            const auto sourceRight = sourceChannels > 1 ? interpolate(1) : sourceLeft;
            const auto absoluteSeconds = blockStart + static_cast<double>(outputBegin + outputOffset) / sampleRate;
            const auto gain = smoothedGain(*loaded,
                absoluteSeconds - clip.startSeconds, outputBegin + outputOffset)
                * loaded->trackGain;
            const auto destination = info.startSample + outputBegin + outputOffset;
            const auto renderedLeft = sourceLeft * gain * leftPan;
            const auto renderedRight = sourceRight * gain * rightPan;
            info.buffer->addSample(0, destination, renderedLeft);
            if (info.buffer->getNumChannels() > 1)
                info.buffer->addSample(1, destination, renderedRight);
            if (loaded->meter != nullptr)
            {
                const auto peak = std::max(std::abs(renderedLeft), std::abs(renderedRight));
                loaded->meter->store(std::max(loaded->meter->load(std::memory_order_relaxed), peak),
                                     std::memory_order_relaxed);
            }
        }
    }

    // A Melodyne project may contain several overlapping elements whose
    // individual gains are valid but whose sum exceeds full scale.  Apply one
    // linked sample envelope (fast attack, slow release) after mixing so block
    // boundaries cannot become gain steps or digital crack/burst artefacts.
    auto limiterGain = masterLimiterGain.load(std::memory_order_relaxed);
    const auto attackMemory = std::exp(-1.0f / static_cast<float>(
        std::max(1.0, sampleRate) * 0.0005));
    const auto releaseMemory = std::exp(-1.0f / static_cast<float>(
        std::max(1.0, sampleRate) * 0.18));
    for (int index = 0; index < info.numSamples; ++index)
    {
        auto linkedPeak = 0.0f;
        for (int channel = 0; channel < info.buffer->getNumChannels(); ++channel)
            linkedPeak = std::max(linkedPeak, std::abs(info.buffer->getSample(
                channel, info.startSample + index)));
        const auto target = linkedPeak > 0.98f ? 0.98f / linkedPeak : 1.0f;
        const auto memory = target < limiterGain ? attackMemory : releaseMemory;
        limiterGain = target + (limiterGain - target) * memory;
        for (int channel = 0; channel < info.buffer->getNumChannels(); ++channel)
        {
            auto value = info.buffer->getSample(channel, info.startSample + index) * limiterGain;
            const auto magnitude = std::abs(value);
            if (magnitude > 0.98f)
                value = std::copysign(0.98f + 0.02f
                    * std::tanh((magnitude - 0.98f) / 0.02f), value);
            info.buffer->setSample(channel, info.startSample + index, value);
        }
    }
    masterLimiterGain.store(limiterGain, std::memory_order_relaxed);

    const auto nextSample = timelineSample.fetch_add(info.numSamples) + info.numSamples;
    const auto reached = static_cast<double>(nextSample) / sampleRate;
    const auto until = playUntilSeconds.load();
    if (reached >= projectDurationSeconds.load() || (until > 0.0 && reached >= until))
        playing.store(false);
    if (!offlineRendering.load(std::memory_order_relaxed)) sendChangeMessage();
}

void AudioEngine::play()
{
    auto duration = projectDurationSeconds.load();
    {
        const juce::ScopedReadLock guard(renderLock);
        if (auditionMode.load() && auditionReader != nullptr)
            duration = static_cast<double>(auditionReader->lengthInSamples) / auditionReader->sampleRate;
    }
    if (duration > 0.0 && position() >= duration)
        setPosition(0.0);
    playing.store(true);
    sendChangeMessage();
}

void AudioEngine::setPlayUntil(double seconds)
{
    playUntilSeconds.store(std::isfinite(seconds) ? std::max(0.0, seconds) : 0.0);
}

void AudioEngine::stop()
{
    playing.store(false);
    masterLimiterGain.store(1.0f, std::memory_order_relaxed);
    {
        const juce::ScopedReadLock guard(renderLock);
        for (const auto& [_, meter] : trackMeters)
            meter->store(0.0f, std::memory_order_relaxed);
    }
    sendChangeMessage();
}

void AudioEngine::setPosition(double seconds)
{
    timelineSample.store(static_cast<juce::int64>(std::max(0.0, seconds) * outputSampleRate.load()));
    if (!auditionMode.load()) renderService.setPlaybackPosition(position());
    sendChangeMessage();
}

double AudioEngine::position() const
{
    return static_cast<double>(timelineSample.load()) / outputSampleRate.load();
}

float AudioEngine::trackPeak(const juce::String& trackId) const
{
    const juce::ScopedReadLock guard(renderLock);
    if (const auto found = trackMeters.find(trackId.toStdString()); found != trackMeters.end())
        return found->second->load(std::memory_order_relaxed);
    return 0.0f;
}

bool AudioEngine::pendingAudioInRange(double start, double end) const
{
    const auto until = playUntilSeconds.load(std::memory_order_relaxed);
    if (until > start) end = std::min(end, until);
    for (const auto& loaded : loadedClips)
    {
        const auto& state = loaded->rendered;
        if (loaded->clip.muted || state == nullptr
            || state->ready.load(std::memory_order_acquire)
            || state->finished.load(std::memory_order_acquire)) continue;
        const auto from = loaded->clip.startSeconds + state->timelineOffsetSeconds;
        const auto to = loaded->clip.startSeconds + loaded->clip.durationSeconds;
        if (from < end && to > start) return true;
    }
    return false;
}

bool AudioEngine::playbackNeedsRender(double lookAheadSeconds) const
{
    const juce::ScopedReadLock guard(renderLock);
    if (auditionMode.load()) return false;
    return pendingAudioInRange(position(), position() + std::max(0.001, lookAheadSeconds));
}

std::optional<double> AudioEngine::renderProgress() const
{
    const juce::ScopedReadLock guard(renderLock);
    int total = 0;
    auto completed = 0.0;
    auto anyUnfinished = false;
    for (const auto& loaded : loadedClips)
        if (loaded->rendered != nullptr)
        {
            ++total;
            const auto done = loaded->rendered->finished.load(std::memory_order_acquire);
            if (!done) anyUnfinished = true;
            completed += done ? 1.0 : static_cast<double>(loaded->rendered->progress.load(
                std::memory_order_acquire));
        }
    // Reporting no progress means finished, so a clip whose progress reached
    // 1.0 before its result was published must not count: callers that wait
    // for this and then export were told to go ahead too early and failed
    // with "pre-render is still running".
    if (total == 0 || (!anyUnfinished && completed >= static_cast<double>(total)))
        return std::nullopt;
    return juce::jlimit(0.0, 0.999, completed / static_cast<double>(total));
}

bool AudioEngine::hasPlayableRenderedAudio() const
{
    const juce::ScopedReadLock guard(renderLock);
    for (const auto& loaded : loadedClips)
        for (const auto& rendered : { loaded->rendered, loaded->fallbackRendered })
            if (rendered != nullptr
                && rendered->ready.load(std::memory_order_acquire)
                && rendered->buffer.getNumSamples() > 0)
                return true;
    return false;
}

bool AudioEngine::hasCurrentRenderedAudio() const
{
    const juce::ScopedReadLock guard(renderLock);
    for (const auto& loaded : loadedClips)
        if (loaded->rendered != nullptr
            && loaded->rendered->ready.load(std::memory_order_acquire)
            && loaded->rendered->buffer.getNumSamples() > 0
            && loaded->rendered->lastAudibleSample >= loaded->rendered->firstAudibleSample)
            return true;
    return false;
}

bool AudioEngine::rewindToFirstPlayableRenderedAudio(double leadInSeconds)
{
    std::optional<double> firstAudibleSeconds;
    {
        const juce::ScopedReadLock guard(renderLock);
        for (const auto& loaded : loadedClips)
        {
            const auto rendered = loaded->rendered != nullptr
                    && loaded->rendered->ready.load(std::memory_order_acquire)
                ? loaded->rendered : loaded->fallbackRendered;
            if (rendered == nullptr
                || !rendered->ready.load(std::memory_order_acquire)
                || rendered->sampleRate <= 0.0
                || rendered->lastAudibleSample < rendered->firstAudibleSample)
                continue;
            const auto absolute = loaded->clip.startSeconds
                + rendered->timelineOffsetSeconds
                + static_cast<double>(rendered->firstAudibleSample) / rendered->sampleRate;
            firstAudibleSeconds = firstAudibleSeconds
                ? std::min(*firstAudibleSeconds, absolute) : absolute;
        }
    }
    if (!firstAudibleSeconds) return false;
    setPosition(std::max(0.0, *firstAudibleSeconds - std::max(0.0, leadInSeconds)));
    return true;
}

juce::String AudioEngine::activeRenderBackends() const
{
    const juce::ScopedReadLock guard(renderLock);
    juce::StringArray names;
    for (const auto& loaded : loadedClips)
        if (loaded->rendered != nullptr
            && loaded->rendered->ready.load(std::memory_order_acquire)
            && loaded->rendered->backend.isNotEmpty())
            names.addIfNotAlreadyThere(loaded->rendered->backend);
    names.sort(true);
    return names.joinIntoString(" + ");
}

juce::String AudioEngine::activeRenderWarnings() const
{
    const juce::ScopedReadLock guard(renderLock);
    juce::StringArray warnings;
    for (const auto& loaded : loadedClips)
        if (loaded->rendered != nullptr
            && loaded->rendered->finished.load(std::memory_order_acquire)
            && loaded->rendered->warning.isNotEmpty())
            warnings.addIfNotAlreadyThere(loaded->rendered->warning);
    return warnings.joinIntoString("; ");
}

juce::StringArray AudioEngine::renderCapabilityWarnings(const ProjectData& project, UtauOutputEngine defaultEngine)
{
    // Native NSF voicebank synthesis shares OTO timing, velocity and STP.
    // HiFisampler implements its own FLAG catalogue; WCSNDM-only names warn.
    // Audio-source backends keep their common pitch/amplitude parameters.
    juce::StringArray warnings;
    for (const auto& track : project.tracks)
    {
        const auto nativeOto = trackUsesVoicebankSynthesis(track) && !trackIsDiffSinger(track)
            && effectiveUtauOutputEngine(track, defaultEngine) == UtauOutputEngine::pcNsfHifigan;
        if (track.accompaniment || (track.pitchAlgorithm == PitchAlgorithm::utau && !nativeOto)) continue;
        auto usesFlags = track.utauGlobalFlags.trim().isNotEmpty();
        auto usesFlagCurve = false;
        auto usesConsonantVelocity = false;
        auto usesTimingOverride = false;
        auto usesStp = false;
        auto usesRegionFlags = false;
        for (const auto& clip : track.clips)
            for (const auto& note : clip.notes)
            {
                usesFlags = usesFlags || note.utauFlags.trim().isNotEmpty();
                usesFlagCurve = usesFlagCurve || note.utauFlagCurveEnabled;
                usesConsonantVelocity = usesConsonantVelocity
                    || (note.utauConsonantVelocity != inheritedUtauConsonantVelocity
                        && note.utauConsonantVelocity != 100);
                // Material P/overlap snapshots are also the canonical native
                // source geometry, not an unsupported UTAU-only override. Only
                // suppress a warning when the stored source markers AGREE;
                // independently edited protocol values still need a warning.
                const auto boundTiming = note.materialAudioId.isNotEmpty() && note.materialRegionId.isNotEmpty()
                    && note.nativeSourceStartSeconds >= 0 && note.nativeSourceEndSeconds > note.nativeSourceStartSeconds
                    && !note.nativeSegments.empty();
                const auto sourceOrigin = note.materialSourceStartSeconds >= 0
                    ? note.materialSourceStartSeconds : note.nativeSourceStartSeconds;
                const auto sourceLength = std::max(0.0, note.nativeSourceEndSeconds - note.nativeSourceStartSeconds);
                const auto projectedP = juce::jlimit(0.0, sourceLength,
                    sourceOrigin + note.utauPreutteranceSeconds - note.nativeSourceStartSeconds);
                const auto pIsSnapshot = boundTiming && std::all_of(note.nativeSegments.begin(),note.nativeSegments.end(),
                    [&](const auto& segment){return std::abs(segment.alignmentSeconds-projectedP)<=1.0e-6;});
                const auto overlapIsSnapshot = boundTiming && std::all_of(note.nativeSegments.begin(),note.nativeSegments.end(),
                    [&](const auto& segment){return std::abs(segment.overlapSeconds-note.utauOverlapSeconds)<=1.0e-6;});
                usesTimingOverride = usesTimingOverride
                    || (note.utauPreutteranceOverrideEnabled && std::abs(note.utauPreutteranceSeconds) > 1.0e-6 && !pIsSnapshot)
                    || (note.utauOverlapOverrideEnabled && std::abs(note.utauOverlapSeconds) > 1.0e-6 && !overlapIsSnapshot);
                usesStp = usesStp || std::abs(note.utauStpSeconds) > 1.0e-6;
                usesRegionFlags = usesRegionFlags || note.utauFlagSplit
                    || note.utauRegionFlags1.isNotEmpty()
                    || note.utauRegionFlags2.isNotEmpty()
                    || note.utauRegionFlags3.isNotEmpty()
                    || note.utauRegionFlags4.isNotEmpty();
            }
        const auto note = [&](const juce::String& feature)
        {
            warnings.addIfNotAlreadyThere("[" + track.name + "] " + feature);
        };
        if (nativeOto) {
            juce::StringArray ignored;
            const auto inspect = [&](const juce::String& text) {
                ignored.addArray(backend::HifisamplerFlags::parse(text).unsupported);
            };
            inspect(track.utauGlobalFlags);
            for (const auto& clip : track.clips) for (const auto& n : clip.notes) {
                inspect(n.utauFlags);
                if (n.utauFlagSplit) for (const auto* value : { &n.utauRegionFlags1, &n.utauRegionFlags2, &n.utauRegionFlags3, &n.utauRegionFlags4 }) inspect(*value);
                if (n.utauFlagCurveEnabled) for (const auto& c : n.utauFlagCurves) {
                    const auto name = c.flag.startsWith("HIFI:") ? c.flag.substring(5) : c.flag;
                    if (!backend::hifiDefinition(name)) ignored.addIfNotAlreadyThere(c.flag);
                }
            }
            ignored.removeDuplicates(false);
            if (!ignored.isEmpty()) note(juce::String::fromUTF8("HiFisampler 不支持这些 FLAG：") + ignored.joinIntoString(", "));
        }
        else if (usesFlags || usesFlagCurve)
            note(juce::String::fromUTF8("UTAU flags 仅在 UTAU 渲染后端生效，当前后端将忽略"));

        if (usesConsonantVelocity && !nativeOto)
            note(juce::String::fromUTF8("辅音速度仅 UTAU 后端生效；其他后端用起音时间映射近似"));
        if (usesTimingOverride && !nativeOto)
            note(juce::String::fromUTF8("先行/交叠覆盖仅 UTAU 后端生效"));
        if (usesStp && !nativeOto)
            note(juce::String::fromUTF8("STP 仅 UTAU 后端生效"));
        if (usesRegionFlags && !nativeOto)
            note(juce::String::fromUTF8("分区 flag 仅 UTAU 后端生效"));
    }
    return warnings;
}

juce::String AudioEngine::componentExportIssue(const ProjectData& project,
    const juce::File& engine, const juce::String& trackId,
    const std::vector<juce::String>& noteIds, juce::Range<double> range,
    const juce::String& auditionTrack, UtauOutputEngine defaultEngine)
{
    const auto anySolo = std::any_of(project.tracks.begin(), project.tracks.end(),
        [](const auto& track) { return track.solo; });
    bool hasSound = false;
    for (const auto& track : project.tracks)
    {
        if (trackId.isNotEmpty() && track.id != trackId) continue;
        if (!trackIsAudible(track.muted, track.solo, anySolo, track.referenceOnly, track.id == auditionTrack)) continue;
        for (const auto& clip : track.clips)
        {
            if (clip.muted || (range.getLength() > 0 &&
                (clip.startSeconds >= range.getEnd() || clip.startSeconds + clip.durationSeconds <= range.getStart()))) continue;
            if (track.accompaniment && clip.sourceFile.existsAsFile())
                return track.name + ": 伴奏轨道直接播放原音，不提供气声 / 非气声合成分量。";
            if (track.pitchAlgorithm == PitchAlgorithm::utau)
            {
                for (const auto& note : clip.notes)
                {
                    if (!noteIds.empty() && std::find(noteIds.begin(), noteIds.end(), note.id) == noteIds.end()) continue;
                    if (backend::isRestLyric(note.label) || (!trackIsDiffSinger(track) && note.label.trim().isEmpty())) continue;
                    hasSound = true;
                    if (!track.compose || backend::DiffSingerRenderer::isVoicebank(track.voicebankDirectory) || note.label.trim().isEmpty()
                        || effectiveUtauOutputEngine(track, defaultEngine) == UtauOutputEngine::pcNsfHifigan
                        || !backend::UtauRenderer::supportsComponentExport(effectiveUtauResampler(track, engine), note.utauFlags + track.utauGlobalFlags))
                        return track.name + ": UTAU 分量导出需要 WCSNDM 的纯 K2 内核（无 u / Mm 混合）；DS 暂不支持。";
                }
            }
            else
            {
                if (!clip.sourceFile.existsAsFile()) continue;
                hasSound = true;
                if (!track.compose || clip.notes.empty() ||
                    (track.pitchAlgorithm != PitchAlgorithm::llsm2 && track.pitchAlgorithm != PitchAlgorithm::world))
                    return track.name + ": 当前调音方式不提供气声 / 非气声分量（支持 LLSM、WORLD）。";
            }
        }
    }
    return hasSound ? juce::String{} : juce::String("当前导出范围没有可分离的人声。");
}

bool AudioEngine::exportWav(const juce::File& file, juce::String& error,
                            const juce::String& trackId,
                            double fromSeconds, double toSeconds, WavExportOptions options)
{
    error.clear();
    if (!options.isValid())
    {
        error = "Invalid WAV export settings";
        return false;
    }
    if (options.component != renderComponent || componentExportError.isNotEmpty())
    {
        error = componentExportError.isNotEmpty() ? componentExportError
            : "Export component does not match the prepared audio";
        return false;
    }
    const auto failure = activeRenderWarnings();
    if (failure.isNotEmpty())
    {
        error = failure;
        return false;
    }
    {
        const juce::ScopedReadLock guard(renderLock);
        for (const auto& loaded : loadedClips)
        {
            if (trackId.isNotEmpty() && loaded->trackId != trackId.toStdString()) continue;
            if (loaded->rendered != nullptr && loaded->rendered->hasEmptyLyricPreview)
            {
                error = "Empty-lyric piano preview is not export audio; prepare with syncProject(project, true)";
                return false;
            }
            if (options.component != WavExportComponent::full
                && (loaded->rendered == nullptr || !loaded->rendered->ready.load(std::memory_order_acquire)))
            {
                // Empty UTAU selection/rest-only clips have no reader or sound.
                if (loaded->reader == nullptr && loaded->rendered == nullptr) continue;
                error = "Separated audio is unavailable; complete voice cannot be substituted";
                return false;
            }
            if (loaded->rendered != nullptr
                && !loaded->rendered->finished.load(std::memory_order_acquire))
            {
                error = "Pre-render is still running";
                return false;
            }
        }
    }

    // Replace an existing export only after its replacement was fully written.
    juce::TemporaryFile temporary(file);
    auto stream = temporary.getFile().createOutputStream();
    if (stream == nullptr || !stream->openedOk())
    {
        error = "Could not create " + file.getFullPathName();
        return false;
    }
    const auto sampleRate = options.sampleRate > 0 ? static_cast<double>(options.sampleRate)
        : juce::jlimit(8'000.0, 192'000.0, outputSampleRate.load());
    juce::WavAudioFormat format;
    auto writer = std::unique_ptr<juce::AudioFormatWriter>(format.createWriterFor(
        stream.get(), sampleRate, static_cast<unsigned int>(options.channels), options.bitDepth, {}, 0));
    if (writer == nullptr)
    {
        error = "Could not create WAV writer";
        return false;
    }
    auto* outputStream = stream.release(); // The writer now owns the stream.

    stop();
    {
        const juce::ScopedWriteLock guard(renderLock);
        exportTrackFilter = trackId.toStdString();
    }
    deviceManager.removeAudioCallback(&sourcePlayer);
    const auto previousPosition = timelineSample.load();
    // The mixer resamples source audio using this rate, not just the WAV header.
    const auto previousSampleRate = outputSampleRate.exchange(sampleRate);
    const auto previousAudition = auditionMode.exchange(false);
    // Playing a selection leaves a stop-here mark behind, and the offline pass
    // runs through the same block callback that honours it: past that moment
    // the transport switched itself off and every remaining block came out
    // empty, so the file was full length with only its opening filled in.
    // An export is not playback and has no business stopping early.
    const auto previousPlayUntil = playUntilSeconds.exchange(0.0);
    offlineRendering.store(true, std::memory_order_release);
    const auto songSeconds = projectDurationSeconds.load();
    const auto fromClamped = juce::jlimit(0.0, std::max(0.0, songSeconds), fromSeconds);
    const auto toClamped = toSeconds > fromClamped
        ? std::min(toSeconds, songSeconds) : songSeconds;
    timelineSample.store(static_cast<juce::int64>(std::llround(fromClamped * sampleRate)));
    masterLimiterGain.store(1.0f, std::memory_order_relaxed);
    playing.store(true);

    constexpr int blockSize = 2048;
    // Always mix in stereo, then average both channels for mono export.
    // Reading only the left channel would lose right-panned instruments.
    juce::AudioBuffer<float> block(2, blockSize);
    const auto totalSamples = static_cast<juce::int64>(std::ceil(
        std::max(0.0, toClamped - fromClamped) * sampleRate));
    auto written = juce::int64(0);
    auto ok = true;
    while (written < totalSamples)
    {
        const auto count = static_cast<int>(std::min<juce::int64>(blockSize, totalSamples - written));
        block.clear();
        juce::AudioSourceChannelInfo info(&block, 0, count);
        getNextAudioBlock(info);
        if (options.channels == 1)
        {
            block.applyGain(0, 0, count, 0.5f);
            block.addFrom(0, 0, block, 1, 0, count, 0.5f);
        }
        if (!writer->writeFromAudioSampleBuffer(block, 0, count))
        {
            ok = false;
            error = "WAV write failed";
            break;
        }
        written += count;
    }

    ok = writer->flush() && ok;
    outputStream->flush();
    ok = outputStream->getStatus().wasOk() && ok;
    if (!ok && error.isEmpty()) error = "WAV flush failed";
    writer.reset();
    playing.store(false);
    {
        const juce::ScopedWriteLock guard(renderLock);
        exportTrackFilter.clear();
    }
    outputSampleRate.store(previousSampleRate);
    timelineSample.store(previousPosition);
    playUntilSeconds.store(previousPlayUntil);
    auditionMode.store(previousAudition);
    offlineRendering.store(false, std::memory_order_release);
    deviceManager.addAudioCallback(&sourcePlayer);
    if (ok && !temporary.overwriteTargetFileWithTemporary())
    {
        ok = false;
        error = "Could not replace " + file.getFullPathName();
    }
    sendChangeMessage();
    return ok;
}
}
