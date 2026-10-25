#include "NativeAudioFocus.h"
#include "AudioEngine.h"
#include "SourceWaveformPreview.h"
#include "TimelineComponent.h"
#include "ClipParts.h"
#include "TrackGainEnvelope.h"
#include "Theme.h"
#include <algorithm>
#include <array>
#include <cmath>

namespace hachi
{
namespace
{
struct GainControlLayout
{
    juce::Rectangle<float> bounds, icon, knob, value;
};

GainControlLayout gainControlLayout(juce::Rectangle<float> clip)
{
    GainControlLayout result;
    // Scale graphics with the lane height, but keep a readable text size.
    // In the shortest lanes share the header row, beside the mute badge.
    const auto compact = clip.getHeight() < 34.0f;
    const auto left = clip.getX() + (compact ? 20.0f : 8.0f);
    const auto available = clip.getRight() - left - 4.0f;
    const auto showValue = available >= 50.0f;
    auto size = std::min(juce::jlimit(7.0f, 18.0f, (clip.getHeight()-17.0f)*0.26f),
        available - (showValue ? 54.0f : 0.0f));
    if (size < 6.0f) size = 0.0f;
    if (!showValue && size == 0.0f) return result;
    const auto textHeight = std::min(14.0f, clip.getHeight()-2.0f);
    const auto height = std::max(size, showValue ? textHeight : 0.0f);
    const auto centreY = compact ? clip.getY()+std::min(8.5f, clip.getHeight()*0.5f)
        : clip.getBottom()-3.0f-height*0.5f;
    const auto iconSize = size*0.8f;
    const auto showIcon = size >= 8.0f
        && available >= size+iconSize+3.0f+(showValue ? 54.0f : 0.0f);
    auto x = left;
    if (showIcon)
    {
        result.icon = {x, centreY-iconSize*0.5f, iconSize, iconSize};
        x += iconSize+3.0f;
    }
    if (size > 0.0f)
    {
        result.knob = {x, centreY-size*0.5f, size, size};
        x += size+(showValue ? 4.0f : 0.0f);
    }
    if (showValue)
    {
        result.value = {x, centreY-textHeight*0.5f, 50.0f, textHeight};
        x += 50.0f;
    }
    result.bounds = juce::Rectangle<float>(left-1.0f, centreY-height*0.5f-1.0f,
        x-left+2.0f, height+2.0f).getIntersection(clip);
    return result;
}

juce::Rectangle<float> gainControlBounds(juce::Rectangle<float> clip)
{
    return gainControlLayout(clip).bounds;
}

juce::String gainValueLabel(float gain)
{
    if (gain <= 0.0f) return juce::String::fromUTF8("-∞");
    const auto db = 20.0f * std::log10(gain);
    const auto rounded = std::round(db * 10.0f) / 10.0f;
    const auto whole = std::abs(rounded - std::round(rounded)) < 0.001f;
    return (rounded >= 0.0f ? "+" : "") + juce::String(rounded == 0.0f ? 0.0f : rounded, whole ? 0 : 1);
}

juce::String gainLabel(float gain)
{
    return (gain <= 0.0f ? juce::String("-inf")
                          : juce::String(20.0f * std::log10(gain), 1)) + " dB";
}

bool hasPreviewNote(const ClipData& clip, const NoteData& note)
{
    return std::isfinite(note.startSeconds) && std::isfinite(note.durationSeconds)
        && std::isfinite(note.midiNote) && note.durationSeconds > 0.0
        && note.startSeconds < clip.durationSeconds
        && note.startSeconds + note.durationSeconds > 0.0;
}
}

TimelineComponent::TimelineComponent(ProjectModel& modelToUse) : model(modelToUse)
{
    formats.registerBasicFormats();
    setWantsKeyboardFocus(true);
    model.addChangeListener(this);
    rebuild();
    startTimerHz(12);
}

TimelineComponent::~TimelineComponent()
{
    stopTimer();
    model.removeChangeListener(this);
}

float TimelineComponent::timeToX(double seconds) const
{
    return static_cast<float>(seconds) * pixelsPerSecond;
}

int TimelineComponent::pixelForSeconds(double seconds) const
{
    return static_cast<int>(std::round(timeToX(seconds)));
}

double TimelineComponent::secondsForPixel(int pixel) const
{
    return std::max(0.0, static_cast<double>(pixel) / pixelsPerSecond);
}

double TimelineComponent::gridQuarterNotes() const
{
    auto text = snapshot.gridDivision.trim().toLowerCase();
    auto dotted = text.endsWithChar('.');
    auto triplet = text.endsWithChar('t');
    if (dotted || triplet) text = text.dropLastCharacters(1);
    const auto slash = text.indexOfChar('/');
    const auto denominator = slash >= 0 ? text.substring(slash + 1).getIntValue() : 16;
    auto quarters = 4.0 / static_cast<double>(std::max(1, denominator));
    if (dotted) quarters *= 1.5;
    if (triplet) quarters *= 2.0 / 3.0;
    return std::max(1.0 / 256.0, quarters);
}

double TimelineComponent::snapToGrid(double seconds) const
{
    const auto step = gridQuarterNotes();
    const auto quarter = snapshot.quarterPositionForSeconds(seconds);
    return std::max(0.0, snapshot.secondsForQuarterPosition(
        std::round(quarter / step) * step));
}

juce::String TimelineComponent::trackIdForPixel(int pixel) const
{
    if (pixel < rulerHeight) return {};
    const auto index = (pixel - rulerHeight) / rowHeight;
    return index >= 0 && index < static_cast<int>(snapshot.tracks.size())
        ? snapshot.tracks[static_cast<std::size_t>(index)].id : juce::String{};
}

void TimelineComponent::setPixelsPerSecond(float value)
{
    // Matches the roll: the two are driven by one slider, and letting them
    // clamp differently put them on different scales past 600, after which
    // nothing that translated between them could be right.
    pixelsPerSecond = juce::jlimit(40.0f, 8000.0f, value);
    rebuild();
}

void TimelineComponent::setRowHeight(float value)
{
    rowHeight = juce::jlimit(40, 220, static_cast<int>(std::round(value)));
    rebuild();
}

void TimelineComponent::setPlayheadSeconds(double seconds)
{
    // Same as the roll: a one pixel line does not need the whole arrangement
    // repainted thirty times a second, and the band has to cover where the
    // line was as well as where it is going.
    if (std::abs(seconds - playheadSeconds) < 1.0e-9) return;
    const auto before = timeToX(playheadSeconds);
    playheadSeconds = seconds;
    const auto after = timeToX(playheadSeconds);
    const auto left = static_cast<int>(std::floor(std::min(before, after))) - 3;
    const auto right = static_cast<int>(std::ceil(std::max(before, after))) + 3;
    repaint(left, 0, std::max(1, right - left), getHeight());
}

void TimelineComponent::changeListenerCallback(juce::ChangeBroadcaster*)
{
    rebuild();
}

void TimelineComponent::timerCallback()
{
    repaint();
}

void TimelineComponent::setShowNativeRenderedWaveforms(bool show)
{
    if(show==showNativeRenderedWaveforms)return;
    showNativeRenderedWaveforms=show;rebuildNativeWaveformHashes();repaint();
}
void TimelineComponent::rebuildNativeWaveformHashes()
{
    nativeClipHashes.clear();if(!showNativeRenderedWaveforms)return;
    for(const auto& track:snapshot.tracks)
        if(track.compose&&!track.accompaniment&&!trackUsesVoicebankSynthesis(track))
            for(const auto& clip:track.clips)for(const auto& part:expandedClipParts(clip))
                nativeClipHashes.emplace(part.id.toStdString(),AudioEngine::nativeClipWaveformHash(part,track));
}

void TimelineComponent::setNativeClipWaveforms(std::shared_ptr<const std::vector<NativeRenderedWaveform>> waveforms)
{
    if(nativeWaveforms==waveforms)return;
    nativeWaveforms=std::move(waveforms);nativePeaksByHash.clear();
    if(nativeWaveforms)for(const auto& waveform:*nativeWaveforms)
        if(waveform.peaks)nativePeaksByHash.try_emplace(waveform.audioHash,waveform.peaks.get());
    repaint();
}

void TimelineComponent::rebuild()
{
    const auto owners = [&](const ProjectData& data) {
        std::vector<juce::String> ids;
        for (const auto& track : data.tracks)
            if (std::any_of(track.clips.begin(), track.clips.end(), [&](const auto& clip) { return isClipSelected(clip.id); }))
                ids.push_back(track.id);
        return ids;
    };
    const auto previousOwners = owners(snapshot);
    snapshot = model.snapshot();
    const auto currentOwners = owners(snapshot);
    const auto ownersChanged = previousOwners != currentOwners;
    if (ownersChanged) selectedTracks = currentOwners;
    if (draggedEnvelopeClip.isNotEmpty())
    {
        const auto* clip = findEnvelopeClip(draggedEnvelopeClip);
        if (clip == nullptr || !sameTrackGainEnvelope(clip->gainEnvelope, envelopeBeforeDrag)
            || clip->startSeconds != draggedEnvelopeStart || clip->durationSeconds != draggedEnvelopeDuration)
            clearEnvelopeDrag();
    }
    if (selectedEnvelopeClip.isNotEmpty() && draggedEnvelopeClip.isEmpty())
    {
        const auto* clip = findEnvelopeClip(selectedEnvelopeClip);
        if (clip == nullptr || std::none_of(clip->gainEnvelope.begin(), clip->gainEnvelope.end(),
            [&](const auto& p) { return std::abs(p.timeSeconds-selectedEnvelopeTime)<1.0e-6; }))
            selectedEnvelopeClip.clear();
    }
    const auto oldClipCount = selectedClips.size(), oldTrackCount = selectedTracks.size();
    std::erase_if(selectedClips, [&](const auto& id)
    {
        for (const auto& track : snapshot.tracks) for (const auto& clip : track.clips)
            if (clip.id == id) return false;
        return true;
    });
    std::erase_if(selectedTracks, [&](const auto& id)
    { return std::none_of(snapshot.tracks.begin(), snapshot.tracks.end(), [&](const auto& t) { return t.id == id; }); });
    const auto selectedStillExists = std::any_of(snapshot.tracks.begin(), snapshot.tracks.end(), [this](const auto& track)
    {
        return std::any_of(track.clips.begin(), track.clips.end(), [this](const auto& clip)
        {
            return clip.id == selectedClip;
        });
    });
    if (!selectedStillExists) selectedClip.clear();
    if (ownersChanged || oldClipCount != selectedClips.size() || oldTrackCount != selectedTracks.size()) notifySelection();
    std::unordered_map<std::string, std::unique_ptr<juce::AudioThumbnail>> next;
    for (const auto& track : snapshot.tracks)
        for (const auto& parent : track.clips)
        for (const auto* source : clipSourceRegions(parent))
        {
            const auto& clip = *source;
            const auto key = clip.sourceFile.getFullPathName().toStdString();
            if (next.contains(key)) continue;
            if (const auto found = thumbnails.find(key); found != thumbnails.end())
            {
                next.emplace(key, std::move(found->second));
                continue;
            }
            auto thumbnail = std::make_unique<juce::AudioThumbnail>(nativeWaveformSamplesPerPeak, formats, thumbnailCache);
            if (clip.sourceFile.existsAsFile())
                thumbnail->setSource(new juce::FileInputSource(clip.sourceFile));
            next.emplace(key, std::move(thumbnail));
        }
    thumbnails = std::move(next);
    rebuildNativeWaveformHashes();
    const auto displayEnd = snapshot.durationSeconds();
    setSize(static_cast<int>(juce::jlimit(470.0, 3.2e7,
        static_cast<double>(timeToX(displayEnd)) + 400.0)),
            rulerHeight + std::max(rowHeight, static_cast<int>(snapshot.tracks.size()) * rowHeight));
    repaint();
}

void TimelineComponent::paint(juce::Graphics& g)
{
    g.fillAll(Palette::base);
    const auto light = Palette::background.getPerceivedBrightness() > 0.5f;
    clipHits.clear();
    g.setColour(Palette::background);
    g.fillRect(0, 0, getWidth(), rulerHeight);
    g.setColour(Palette::border);
    g.drawHorizontalLine(rulerHeight - 1, 0.0f, static_cast<float>(getWidth()));
    const auto gridStep = gridQuarterNotes();
    const auto firstQuarter = snapshot.quarterPositionForSeconds(0.0);
    const auto firstTick = static_cast<int>(std::floor(firstQuarter / gridStep)) - 1;
    const auto barQuarters = static_cast<double>(std::max(1, snapshot.numerator))
        * 4.0 / static_cast<double>(std::max(1, snapshot.denominator));
    for (int tick = firstTick;; ++tick)
    {
        const auto quarter = static_cast<double>(tick) * gridStep;
        const auto seconds = snapshot.secondsForQuarterPosition(quarter);
        const auto x = timeToX(seconds);
        if (x > static_cast<float>(getWidth())) break;
        if (x < 0.0f) continue;
        const auto isBeat = std::abs(quarter - std::round(quarter)) < 1.0e-6;
        const auto bar = quarter / barQuarters;
        const auto isBar = std::abs(bar - std::round(bar)) < 1.0e-6;
        g.setColour(isBar ? Palette::grid.brighter(0.28f)
                   : isBeat ? Palette::grid.withAlpha(0.62f) : Palette::grid.withAlpha(0.30f));
        g.drawVerticalLine(static_cast<int>(x), 0.0f, static_cast<float>(getHeight()));
        if (isBar)
        {
            g.setColour(Palette::textMuted);
            g.setFont(10.0f);
            g.drawText(juce::String(static_cast<int>(std::llround(bar)) + 1) + ".1",
                       static_cast<int>(x) + 4, 3, 42, 16, juce::Justification::left);
        }
    }

    for (const auto& change : snapshot.tempoChanges)
    {
        const auto x = timeToX(snapshot.secondsForQuarterPosition(change.quarterPosition));
        if (x < 0.0f || x > getWidth()) continue;
        g.setColour(juce::Colour(0xffffa94d));
        g.drawVerticalLine(static_cast<int>(x), 0.0f, static_cast<float>(getHeight()));
        g.setFont(9.5f);
        if (light) g.setColour(Palette::contrastingText(Palette::background));
        g.drawText(juce::String(change.bpm, 2).trimCharactersAtEnd("0").trimCharactersAtEnd(".")
                + " BPM", static_cast<int>(x) + 4, 3, 66, 16,
            juce::Justification::centredLeft, false);
    }

    // One pitch scale across the arrangement keeps the lead and harmonies
    // comparable, including clips that contain only a single repeated pitch.
    auto lowestPitch = 127.0f, highestPitch = 0.0f;
    for (const auto& track : snapshot.tracks)
        if (track.compose && !track.accompaniment)
            for (const auto& clip : track.clips)
                for (const auto& note : clip.notes)
                    if (hasPreviewNote(clip, note))
                    {
                        const auto pitch = juce::jlimit(0.0f, 127.0f, note.midiNote);
                        lowestPitch = std::min(lowestPitch, pitch);
                        highestPitch = std::max(highestPitch, pitch);
                    }
    if (lowestPitch > highestPitch) { lowestPitch = 60.0f; highestPitch = 72.0f; }
    const auto pitchCentre = (lowestPitch + highestPitch) * 0.5f;
    const auto pitchHalfRange = std::max(6.0f, (highestPitch-lowestPitch)*0.5f + 2.0f);
    const auto pitchTop = pitchCentre + pitchHalfRange;
    const auto pitchSpan = 2.0f * pitchHalfRange;

    // Bucket previews into their destination lanes, drawing moving regions
    // last so a destination region cannot obscure the drag preview.
    std::vector<std::vector<const ClipData*>> displayedClips(snapshot.tracks.size());
    for (const auto movingPass : { false, true })
        for (std::size_t source = 0; source < snapshot.tracks.size(); ++source)
            for (const auto& clip : snapshot.tracks[source].clips) {
                const auto moving = dragMode == DragMode::move && draggedClip.isNotEmpty() && isClipSelected(clip.id);
                if (moving != movingPass) continue;
                const auto destination = static_cast<int>(source)+(moving ? draggedTrackDelta : 0);
                if (destination >= 0 && destination < static_cast<int>(displayedClips.size()))
                    displayedClips[static_cast<std::size_t>(destination)].push_back(&clip);
            }
    for (std::size_t trackIndex = 0; trackIndex < snapshot.tracks.size(); ++trackIndex)
    {
        const auto& track = snapshot.tracks[trackIndex];
        const auto row = juce::Rectangle<int>(0, rulerHeight + static_cast<int>(trackIndex) * rowHeight,
                                               getWidth(), rowHeight);
        g.setColour(Palette::grid);
        g.drawHorizontalLine(row.getBottom() - 1, 0.0f, static_cast<float>(getWidth()));
        const auto colour = Palette::trackColour(trackIndex);
        if (std::find(selectedTracks.begin(), selectedTracks.end(), track.id) != selectedTracks.end())
        {
            g.setColour(Palette::accent.withAlpha(0.06f));
            g.fillRect(row);
        }
        auto shown=track;shown.clips.clear();for(const auto* c:displayedClips[trackIndex])
        {auto value=*c;if(dragMode==DragMode::move&&isClipSelected(value.id))value.startSeconds+=draggedClipPreviewStart-draggedClipStart;shown.clips.push_back(std::move(value));}
        const auto opacity=[&](const auto* c){for(const auto& value:shown.clips)if(value.id==c->id)return nativeOverlapFocusOpacity(shown,value,nativeFocusedNote,selectedClip);return 1.f;};
        std::stable_partition(displayedClips[trackIndex].begin(),displayedClips[trackIndex].end(),[&](const auto* c){return opacity(c)<1.f;});
        for (const auto* displayedClip : displayedClips[trackIndex])
        {
            const auto& clip = *displayedClip;
            const NativeAudioFocusLayer focus(g,opacity(displayedClip));
            const auto showNotes = track.compose && !track.accompaniment && !clip.notes.empty();
            const auto moving = dragMode == DragMode::move && draggedClip.isNotEmpty() && isClipSelected(clip.id);
            const auto displayStart = moving ? clip.startSeconds + draggedClipPreviewStart - draggedClipStart
                : clip.id == draggedClip ? draggedClipPreviewStart : clip.startSeconds;
            const auto displayDuration = clip.id == draggedClip ? draggedClipPreviewDuration
                                                                 : clip.durationSeconds;
            const auto contentStart = clip.startSeconds + clip.audioStartSeconds
                + (moving ? displayStart - clip.startSeconds : 0.0);
            const auto contentDuration = clip.audioLength();
            auto displayFadeIn = clip.fadeInSeconds;
            auto displayFadeOut = clip.fadeOutSeconds;
            if (clip.id == draggedClip
                && (dragMode == DragMode::fadeIn || dragMode == DragMode::fadeOut))
            {
                displayFadeIn = draggedClipPreviewFadeIn;
                displayFadeOut = draggedClipPreviewFadeOut;
            }
            auto bounds = juce::Rectangle<float>(timeToX(displayStart),
                                                  static_cast<float>(row.getY() + 19),
                                                  std::max(4.0f, static_cast<float>(displayDuration)
                                                                      * pixelsPerSecond),
                                                  static_cast<float>(rowHeight - 25));
            clipHits.push_back({ clip.id, bounds, clip.startSeconds, clip.durationSeconds,
                                 clip.fadeInSeconds, clip.fadeOutSeconds, clip.muted,
                                 clip.startSeconds + clip.audioStartSeconds, contentDuration, clip.gain });
            const auto rowSurface = std::find(selectedTracks.begin(), selectedTracks.end(), track.id) != selectedTracks.end()
                ? Palette::base.overlaidWith(Palette::accent.withAlpha(0.06f)) : Palette::base;
            const auto clipInk = track.muted || clip.muted ? Palette::clipBackground.withAlpha(0.55f)
                                                         : Palette::clipBackground;
            const auto clipSurface = rowSurface.overlaidWith(clipInk);
            g.setColour(clipInk);
            g.fillRect(bounds);
            g.setColour(light ? Palette::noteEdge : colour.withAlpha(0.82f));
            g.drawRect(bounds, 1.0f);
            if (isClipSelected(clip.id))
            {
                g.setColour(Palette::text.withAlpha(0.92f));
                g.drawRect(bounds.reduced(1.0f), 2.0f);
            }

            const auto nativePreview = track.compose && !track.accompaniment && !trackUsesVoicebankSynthesis(track);
            const auto nativeParts = nativePreview ? expandedClipParts(clip) : std::vector<ClipData>{};
            std::vector<const ClipData*> sourceParts;
            if (nativePreview) for (const auto& part : nativeParts) sourceParts.push_back(&part);
            else sourceParts = clipSourceRegions(clip);
            for (const auto* part : sourceParts)
            if (const auto found = thumbnails.find(part->sourceFile.getFullPathName().toStdString()); found != thumbnails.end())
            {
                g.setColour(light ? Palette::heading.withAlpha(showNotes ? 0.32f : 0.85f)
                    : colour.brighter(0.65f).withAlpha(
                    (track.muted || clip.muted || part->muted ? 0.25f : 0.78f)
                        * (showNotes ? 0.22f : 1.0f)));
                // Keep the editor visually consistent for mono and stereo
                // sources.  Playback still uses every source channel; only the
                // compact waveform lane displays channel 1.
                const auto partStart = part->startSeconds + part->audioStartSeconds
                    + (nativePreview || clip.parts.empty() ? 0.0 : clip.startSeconds)
                    + (moving ? displayStart - clip.startSeconds : 0.0);
                const auto partDuration = part->audioLength();
                if (partDuration > 1.0e-9)
                {
                    const juce::Graphics::ScopedSaveState save(g);
                    g.reduceClipRegion(bounds.withTrimmedTop(17.0f).toNearestInt());
                    const auto audioBounds = bounds.withX(timeToX(partStart))
                        .withWidth(timeToX(partDuration)).withTrimmedTop(17.0f);
                    if (nativePreview)
                    {
                        const auto audio = nativeAudioPreviewClip(*part);
                        const NativeRenderedPeaks* rendered=nullptr;
                        if(showNativeRenderedWaveforms)
                            if(const auto hash=nativeClipHashes.find(part->id.toStdString());hash!=nativeClipHashes.end())
                                if(const auto peaks=nativePeaksByHash.find(hash->second);peaks!=nativePeaksByHash.end())rendered=peaks->second;
                        const auto opacity=track.muted||clip.muted||part->muted?.30f:(showNotes?.68f:.88f);
                        if(rendered)drawNativeRenderedWaveform(g,*rendered,audio.durationSeconds,audioBounds,opacity);
                        else drawNativeSourceWaveform(g,*found->second,audio,nativeSourceTimeMap(audio),audioBounds,opacity);
                    }
                    else found->second->drawChannel(g, audioBounds.toNearestInt(), part->sourceOffsetSeconds,
                        part->sourceOffsetSeconds + (track.accompaniment ? partDuration
                            : (part->sourceDurationSeconds > 1.0e-9 ? part->sourceDurationSeconds : partDuration)), 0, 1.0f);
                }
            }
            if (showNotes)
            {
                // Keep full-height previews clear of the gain footer. Compact
                // lanes use the remaining body; controls are painted on top.
                const auto noteArea = bounds.withTrimmedTop(19.0f)
                    .withTrimmedBottom(bounds.getHeight() >= 58.0f ? 26.0f : 3.0f);
                if (noteArea.getHeight() >= 2.0f)
                {
                    const juce::Graphics::ScopedSaveState save(g);
                    g.reduceClipRegion(noteArea.toNearestInt());
                    const auto visible = g.getClipBounds().toFloat();
                    const auto noteHeight = juce::jlimit(2.5f, 6.0f, noteArea.getHeight()/pitchSpan*0.85f);
                    // Notes are already parent-local (also in merged clips).
                    // Trimming moves only the boundary, while moving shifts the content.
                    const auto noteOrigin = clip.startSeconds + (moving ? displayStart-clip.startSeconds : 0.0);
                    for (const auto& note : clip.notes)
                    {
                        if (!hasPreviewNote(clip, note)) continue;
                        const auto start = std::max(displayStart, noteOrigin + note.startSeconds);
                        const auto end = std::min(displayStart + displayDuration,
                            noteOrigin + note.startSeconds + note.durationSeconds);
                        if (end <= start) continue;
                        const auto left = timeToX(start), right = timeToX(end);
                        if (right < visible.getX() || left > visible.getRight()) continue;
                        const auto pitch = juce::jlimit(0.0f, 127.0f, note.midiNote);
                        const auto y = noteArea.getY() + (pitchTop-pitch)/pitchSpan*noteArea.getHeight();
                        const juce::Rectangle<float> noteBounds(left, y-noteHeight*0.5f,
                            std::min(std::max(1.0f, right-left), bounds.getRight()-left), noteHeight);
                        const auto owner = std::find_if(clip.parts.begin(), clip.parts.end(),
                            [&](const auto& part) { return part.id == note.clipPartId; });
                        const auto muted = track.muted || clip.muted
                            || (owner != clip.parts.end() && owner->muted);
                        g.setColour(light ? Palette::noteEdge : colour.brighter(0.9f).withAlpha(muted ? 0.34f : 0.96f));
                        g.fillRect(noteBounds);
                        // A fine edge distinguishes consecutive notes at the same pitch.
                        g.setColour(Palette::clipBackground.withAlpha(0.72f));
                        g.drawRect(noteBounds, 0.5f);
                    }
                }
            }
            if (contentDuration > 0.0)
            {
            const juce::Graphics::ScopedSaveState save(g);
            g.reduceClipRegion(bounds.toNearestInt());
            const auto waveformTop = bounds.getY() + 19.0f;
            const auto fadeInX = timeToX(contentStart + displayFadeIn);
            const auto fadeOutX = timeToX(contentStart + contentDuration - displayFadeOut);
            if (displayFadeIn > 0.0)
            {
                g.setColour(Palette::text.withAlpha(0.7f));
                g.drawLine(timeToX(contentStart), bounds.getBottom(),
                           fadeInX, waveformTop, 1.0f);
            }
            if (displayFadeOut > 0.0)
            {
                g.setColour(Palette::text.withAlpha(0.7f));
                g.drawLine(fadeOutX, waveformTop,
                           timeToX(contentStart + contentDuration), bounds.getBottom(), 1.0f);
            }
            g.setColour(light ? Palette::noteEdge : colour.brighter(0.72f).withAlpha(
                isClipSelected(clip.id) ? 0.95f : 0.52f));
            g.fillEllipse(fadeInX - 3.5f, waveformTop - 3.5f, 7.0f, 7.0f);
            g.fillEllipse(fadeOutX - 3.5f, waveformTop - 3.5f, 7.0f, 7.0f);
            }
            g.setColour(Palette::text);
            g.setColour(Palette::panel.withAlpha(0.82f));
            g.fillRect(bounds.toNearestInt().withHeight(17));
            g.setColour(clip.muted ? Palette::noteFill : colour.brighter(0.5f));
            g.fillRoundedRectangle(bounds.getX() + 2.0f, bounds.getY() + 2.0f, 14.0f, 13.0f, 2.0f);
            g.setColour(light ? Palette::contrastingText(clip.muted ? Palette::noteFill : colour.brighter(0.5f)) : Palette::panel);
            g.setFont(9.0f);
            g.drawText("M", static_cast<int>(bounds.getX() + 2.0f), static_cast<int>(bounds.getY() + 1.0f),
                       14, 14, juce::Justification::centred);
            g.setColour(light ? Palette::contrastingText(clipSurface.overlaidWith(Palette::panel.withAlpha(0.82f))) : Palette::text);
            g.setFont(10.0f);
            const auto displayGain = clip.id == draggedClip && dragMode == DragMode::gain
                ? draggedClipPreviewGain : clip.gain;
            const auto gainControl = gainControlLayout(bounds);
            const auto knob = gainControl.knob;
            g.drawText(clipHeaderText(track, clip, displayGain),
                       bounds.toNearestInt().withTrimmedLeft(bounds.getHeight() < 34.0f && !gainControl.bounds.isEmpty()
                           ? static_cast<int>(std::ceil(gainControl.bounds.getRight()-bounds.getX()+3.0f)) : 19).withHeight(17),
                       juce::Justification::centredLeft, true);
            g.setColour(light ? Palette::noteEdge : colour.brighter(0.65f));
            juce::Path leftHandle;
            leftHandle.addTriangle(bounds.getX(), bounds.getY(), bounds.getX() + 7.0f, bounds.getY(),
                                   bounds.getX(), bounds.getY() + 7.0f);
            g.fillPath(leftHandle);
            juce::Path rightHandle;
            rightHandle.addTriangle(bounds.getRight(), bounds.getY(), bounds.getRight() - 7.0f, bounds.getY(),
                                    bounds.getRight(), bounds.getY() + 7.0f);
            g.fillPath(rightHandle);
            // Visible side grips distinguish boundary edits from dragging the body.
            g.fillRoundedRectangle(bounds.getX() + 2.0f, bounds.getCentreY() - 9.0f, 3.0f, 18.0f, 1.5f);
            g.fillRoundedRectangle(bounds.getRight() - 5.0f, bounds.getCentreY() - 9.0f, 3.0f, 18.0f, 1.5f);
            if (!gainControl.bounds.isEmpty())
            {
                const juce::Graphics::ScopedSaveState save(g);
                g.reduceClipRegion(bounds.toNearestInt());
                const auto grey = light ? Palette::contrastingText(clipSurface, true) : Palette::text.withAlpha(0.56f);
                g.setColour(grey);
                if (!gainControl.icon.isEmpty())
                {
                    // Static ascending bars identify gain, not a live level meter.
                    const auto baseline = gainControl.icon.getBottom();
                    for (int bar=0; bar<8; ++bar)
                    {
                        const auto x = gainControl.icon.getX()+static_cast<float>(bar)*gainControl.icon.getWidth()/8.0f;
                        const auto barHeight = gainControl.icon.getHeight()*static_cast<float>(bar+1)/8.0f;
                        g.drawLine(x, baseline, x, baseline-barHeight, std::max(0.7f, gainControl.icon.getWidth()*0.055f));
                    }
                }
                if (!knob.isEmpty())
                {
                    const auto centre = knob.getCentre();
                    const auto inset = std::max(0.7f, knob.getWidth()*0.07f);
                    const auto radius = knob.getWidth() * 0.5f - inset;
                    const auto db = juce::Decibels::gainToDecibels(displayGain, -60.0f);
                    // Unity gain is twelve o'clock; attenuation and boost turn left/right.
                    const auto angle = juce::MathConstants<float>::pi * 0.75f
                        * juce::jlimit(-1.0f, 1.0f, db / (db < 0.0f ? 60.0f : 12.0f));
                    const auto stroke = juce::jlimit(0.9f, 1.8f, knob.getWidth()*0.1f);
                    g.drawEllipse(knob.reduced(inset), stroke);
                    g.drawLine(centre.x, centre.y, centre.x + std::sin(angle)*radius*0.75f,
                        centre.y - std::cos(angle)*radius*0.75f, stroke);
                    }
                if (!gainControl.value.isEmpty())
                {
                    const auto value = gainValueLabel(displayGain);
                    const juce::Font font(12.0f);
                    const auto valueWidth = font.getStringWidthFloat(value)+1.0f;
                    const auto valueBounds = gainControl.value.withWidth(valueWidth);
                    // Only the small readout has a backing; graphics leave the waveform visible.
                    g.setColour(Palette::clipBackground.withAlpha(0.88f));
                    g.fillRect(gainControl.value.withWidth(std::min(50.0f,valueWidth+22.0f)));
                    g.setFont(font);
                    g.setColour(light ? Palette::contrastingText(clipSurface.overlaidWith(Palette::clipBackground.withAlpha(0.88f))) : juce::Colour(0xff3a9dff));
                    g.drawText(value, valueBounds, juce::Justification::centredLeft, false);
                    g.setColour(grey);
                    g.drawText("dB", valueBounds.withX(valueBounds.getRight()+4.0f).withWidth(18.0f),
                        juce::Justification::centredLeft, false);
                }
            }
            paintClipEnvelope(g,clip,track.muted,trackIndex);
        }
        if(trackShowsAllNativeRegions(track))
        {
            const auto windows=nativeAudioWindows(shown);
            for(std::size_t i=0;i<windows.size();++i)for(std::size_t j=i+1;j<windows.size();++j)
            {
                const auto first=std::max(windows[i].start,windows[j].start),last=std::min(windows[i].end,windows[j].end);if(last-first<=1.e-7)continue;
                const juce::Rectangle<float> band(timeToX(first),static_cast<float>(row.getY()+19),timeToX(last)-timeToX(first),static_cast<float>(rowHeight-22));
                const juce::Graphics::ScopedSaveState save(g);g.reduceClipRegion(band.toNearestInt());g.setColour(juce::Colour(0xffffbf69).withAlpha(.16f));g.fillRect(band);
                g.setColour(juce::Colour(0xffffbf69).withAlpha(.7f));for(auto x=band.getX()-band.getHeight();x<band.getRight();x+=14)g.drawLine(x,band.getBottom(),x+band.getHeight(),band.getY(),1.f);
                g.drawRect(band,1.3f);g.setFont(10.f);g.drawText(nativeOverlapLabel+" "+juce::String((last-first)*1000,0)+" ms",band.withHeight(18),juce::Justification::centred);
            }
        }
    }

    g.setColour(Palette::playhead);
    g.drawVerticalLine(static_cast<int>(timeToX(playheadSeconds)), 0.0f, static_cast<float>(getHeight()));
    if (marqueeActive)
    {
        g.setColour(Palette::accent.withAlpha(0.16f));
        g.fillRect(marqueeBounds);
        g.setColour(Palette::accent.withAlpha(0.9f));
        g.drawRect(marqueeBounds, 1.0f);
    }
}

bool TimelineComponent::isClipSelected(const juce::String& id) const
{
    return std::find(selectedClips.begin(), selectedClips.end(), id) != selectedClips.end();
}

void TimelineComponent::notifySelection()
{
    if (!isClipSelected(selectedClip)) selectedClip = selectedClips.empty() ? juce::String{} : selectedClips.front();
    if (onClipSelected) onClipSelected(selectedClip);
    if (onTracksSelected) onTracksSelected(selectedTracks);
    repaint();
}

void TimelineComponent::setSelectedClips(std::vector<juce::String> ids, bool notify)
{
    nativeFocusedNote.clear();
    selectedClips.clear(); selectedTracks.clear();
    for (const auto& track : snapshot.tracks)
    {
        for (const auto& clip : track.clips)
            if (std::find(ids.begin(), ids.end(), clip.id) != ids.end()) selectedClips.push_back(clip.id);
        if (std::any_of(track.clips.begin(), track.clips.end(), [&](const auto& c) { return isClipSelected(c.id); }))
            selectedTracks.push_back(track.id);
    }
    if (!isClipSelected(selectedClip)) selectedClip = selectedClips.empty() ? juce::String{} : selectedClips.front();
    if (notify) notifySelection();
    repaint();
}

void TimelineComponent::beginMarquee(const juce::MouseEvent& event)
{
    clearEnvelopeDrag(); selectedEnvelopeClip.clear();
    draggedClip.clear(); dragMode = DragMode::none;
    marqueePending = true; marqueeActive = false;
    marqueeAnchor = event.position; marqueeBounds = {};
    marqueeAdditive = event.mods.isShiftDown() || event.mods.isCommandDown();
    marqueeBaseClips = selectedClips; marqueeBaseTracks = selectedTracks;
    if (!marqueeAdditive) setSelectedClips({});
    beginDragAutoRepeat(30);
}

void TimelineComponent::updateMarquee(juce::Point<float> position)
{
    marqueeBounds = juce::Rectangle<float>(marqueeAnchor, position);
    selectedClips = marqueeAdditive ? marqueeBaseClips : std::vector<juce::String>{};
    selectedTracks = marqueeAdditive ? marqueeBaseTracks : std::vector<juce::String>{};
    for (std::size_t i = 0; i < snapshot.tracks.size(); ++i)
    {
        const auto& track = snapshot.tracks[i];
        const auto row = juce::Rectangle<float>(0, static_cast<float>(rulerHeight + i * rowHeight),
            static_cast<float>(getWidth()), static_cast<float>(rowHeight));
        if (!marqueeBounds.intersects(row)) continue;
        if (std::find(selectedTracks.begin(), selectedTracks.end(), track.id) == selectedTracks.end())
            selectedTracks.push_back(track.id);
        for (const auto& clip : track.clips)
        {
            const juce::Rectangle<float> bounds(timeToX(clip.startSeconds), row.getY()+19,
                std::max(4.0f,timeToX(clip.durationSeconds)), static_cast<float>(rowHeight-25));
            if (marqueeBounds.intersects(bounds) && !isClipSelected(clip.id)) selectedClips.push_back(clip.id);
        }
    }
    notifySelection();
}

bool TimelineComponent::keyPressed(const juce::KeyPress& key)
{
    if (key.isKeyCode(juce::KeyPress::escapeKey))
    {
        if (draggedEnvelopeClip.isNotEmpty() || selectedEnvelopeClip.isNotEmpty())
        { clearEnvelopeDrag(); selectedEnvelopeClip.clear(); repaint(); return true; }
        if (marqueePending)
        { selectedClips = marqueeBaseClips; selectedTracks = marqueeBaseTracks; }
        else if (draggedClip.isEmpty()) { selectedClips.clear(); selectedTracks.clear(); }
        marqueePending = marqueeActive = false; draggedClip.clear(); dragMode = DragMode::none;
        setMouseCursor(juce::MouseCursor::NormalCursor);
        notifySelection(); return true;
    }
    if (key.getModifiers().isCommandDown() && key.getKeyCode() == 'A')
    {
        selectedEnvelopeClip.clear();
        std::vector<juce::String> all;
        for (const auto& track : snapshot.tracks) for (const auto& clip : track.clips) all.push_back(clip.id);
        setSelectedClips(std::move(all)); return true;
    }
    if (key.isKeyCode(juce::KeyPress::deleteKey) || key.isKeyCode(juce::KeyPress::backspaceKey))
    {
        if (selectedEnvelopeClip.isNotEmpty())
        { editEnvelopePoint(selectedEnvelopeClip, selectedEnvelopeTime, 1); return true; }
        if (!selectedClips.empty() && onDeleteSelected) onDeleteSelected();
        return true; // Never delete an old note selection in the lower pane.
    }
    return false;
}

std::optional<TimelineComponent::Anchor> TimelineComponent::pointerAnchor() const
{
    return hoverAnchor;
}

void TimelineComponent::rememberPointer(const juce::MouseEvent& event)
{
    const auto trackId = trackIdForPixel(event.y);
    if (trackId.isEmpty())
    {
        hoverAnchor.reset();
        return;
    }
    hoverAnchor = Anchor { trackId, std::max(0.0,
        static_cast<double>(event.position.x) / pixelsPerSecond) };
}

juce::String TimelineComponent::clipHeaderText(const TrackData& track, const ClipData& clip, float gain) const
{
    const auto engine = outputEngineNameProvider ? outputEngineNameProvider(track) : juce::String{};
    return (engine.isEmpty() ? juce::String{} : "[" + engine + "]  ")
        + clip.sourceFile.getFileNameWithoutExtension()
        + (clip.parts.empty() ? juce::String{} : " +" + juce::String(static_cast<int>(clip.parts.size())-1))
        + "  " + gainLabel(gain);
}

void TimelineComponent::mouseMove(const juce::MouseEvent& event)
{
    rememberPointer(event);
    setTooltip({});
    for (const auto& hit : clipHits)
        if (hit.bounds.contains(event.position) && event.position.y < hit.bounds.getY()+17.0f)
            for (const auto& track : snapshot.tracks) for (const auto& clip : track.clips)
                if (clip.id == hit.id) { setTooltip(clipHeaderText(track,clip,clip.gain)); return; }
    if (const auto hit = envelopeHit(event.position, true))
    {
        const auto* clip = findEnvelopeClip(hit->clipId);
        if (clip == nullptr) return;
        const auto& points = displayedEnvelope(*clip);
        const auto db = hit->pointIndex >= 0 ? points[static_cast<std::size_t>(hit->pointIndex)].gainDb
            : trackGainEnvelopeDbAt(points, static_cast<double>(event.position.x)/pixelsPerSecond-clip->startSeconds);
        setTooltip(gainLabel(juce::Decibels::decibelsToGain(db,-60.0f))+"\n"+envelopeHint);
        setMouseCursor(hit->pointIndex >= 0 ? juce::MouseCursor::DraggingHandCursor : juce::MouseCursor::PointingHandCursor);
        return;
    }
    for (auto it = clipHits.rbegin(); it != clipHits.rend(); ++it)
        if (it->bounds.expanded(4.0f, 0.0f).contains(event.position))
        {
            if (gainControlBounds(it->bounds).contains(event.position))
            {
                setTooltip(gainLabel(it->gain) + "\n" + gainKnobHint);
                setMouseCursor(juce::MouseCursor::UpDownResizeCursor);
                return;
            }
            const auto muteBounds = juce::Rectangle<float>(it->bounds.getX() + 2.0f,
                it->bounds.getY() + 2.0f, 14.0f, 13.0f);
            if (muteBounds.contains(event.position))
            {
                setMouseCursor(juce::MouseCursor::PointingHandCursor);
                return;
            }
            const auto waveformTop = it->bounds.getY() + 19.0f;
            const auto fadeIn = juce::Point<float>(timeToX(it->audioStartSeconds + it->fadeInSeconds), waveformTop);
            const auto fadeOut = juce::Point<float>(timeToX(it->audioStartSeconds + it->audioDurationSeconds - it->fadeOutSeconds), waveformTop);
            if (event.position.getDistanceFrom(fadeIn) <= 5.0f
                || event.position.getDistanceFrom(fadeOut) <= 5.0f)
            {
                setMouseCursor(juce::MouseCursor::LeftRightResizeCursor);
                return;
            }
            const auto grip = std::min(14.0f, it->bounds.getWidth() * 0.25f);
            const auto onHandle = event.position.x <= it->bounds.getX() + grip
                || event.position.x >= it->bounds.getRight() - grip;
            setMouseCursor(onHandle ? juce::MouseCursor::LeftRightResizeCursor
                                    : juce::MouseCursor::NormalCursor);
            return;
        }
    setMouseCursor(juce::MouseCursor::NormalCursor);
}

void TimelineComponent::mouseExit(const juce::MouseEvent&)
{
    // Off the lanes there is no track to paste onto.
    hoverAnchor.reset();
    setTooltip({});
    if (draggedClip.isEmpty()) setMouseCursor(juce::MouseCursor::NormalCursor);
}

void TimelineComponent::mouseDoubleClick(const juce::MouseEvent& event)
{
    if (event.mods.isPopupMenu()) return;
    if (const auto hit = envelopeHit(event.position, true))
    {
        draggedClip.clear(); dragMode = DragMode::none;
        marqueePending = marqueeActive = false;
        clearEnvelopeDrag();
        const auto* clip = findEnvelopeClip(hit->clipId);
        if (clip == nullptr) return;
        auto points = clip->gainEnvelope;
        auto time = juce::jlimit(0.0,clip->durationSeconds,static_cast<double>(event.position.x)/pixelsPerSecond-clip->startSeconds);
        if (hit->pointIndex >= 0) time = points[static_cast<std::size_t>(hit->pointIndex)].timeSeconds;
        else
        {
            const auto db = trackGainEnvelopeDbAt(points,time);
            if (points.empty()) points = {{0.0,0.0f},{clip->durationSeconds,0.0f}};
            points.push_back({time,db});
            model.setClipGainEnvelope(hit->clipId,std::move(points));
        }
        selectedEnvelopeClip = hit->clipId; selectedEnvelopeTime = time;
        repaint(); return;
    }
    for (auto it = clipHits.rbegin(); it != clipHits.rend(); ++it)
        if (it->bounds.contains(event.position))
        {
            const auto muteBounds = juce::Rectangle<float>(it->bounds.getX() + 2.0f,
                it->bounds.getY() + 2.0f, 14.0f, 13.0f);
            if (muteBounds.contains(event.position)) return;
            draggedClip.clear();
            dragMode = DragMode::none;
            if (gainControlBounds(it->bounds).contains(event.position))
            {
                model.setClipGain(it->id, 1.0f);
                repaint();
                return;
            }
            setSelectedClips({it->id});
            if (onClipGainRequested) onClipGainRequested(selectedClip);
            repaint();
            return;
    }
}

void TimelineComponent::showTempoMenu(double quarterPosition,
                                      juce::Point<int> screenPosition)
{
    juce::PopupMenu menu;
    menu.addItem(1, juce::String::fromUTF8("改变曲速…"));
    juce::Component::SafePointer<TimelineComponent> safe(this);
    menu.showMenuAsync(juce::PopupMenu::Options().withTargetScreenArea(
        juce::Rectangle<int>(screenPosition.x, screenPosition.y, 1, 1)),
        [safe, quarterPosition](int result)
        {
            if (safe == nullptr || result != 1) return;
            juce::MessageManager::callAsync([safe, quarterPosition]
            {
                if (safe != nullptr) safe->showTempoDialog(quarterPosition);
            });
        });
}

void TimelineComponent::showTempoDialog(double quarterPosition)
{
    const auto initialTempo = snapshot.tempoAtQuarterPosition(quarterPosition);
    auto* dialog = new juce::AlertWindow(
        juce::String::fromUTF8("改变曲速"),
        juce::String::fromUTF8("从当前四分之一小节开始使用新的曲速。\n"
            "同步修改音符：按曲速调整音符位置和长度。\n"
            "不修改音符：保留音符位置和长度，只修改曲速。"),
        juce::MessageBoxIconType::NoIcon);
    dialog->addTextEditor("bpm", juce::String(initialTempo, 2),
                          juce::String::fromUTF8("BPM（20–400）"));
    dialog->addComboBox("noteTiming", { juce::String::fromUTF8("同步修改音符"),
        juce::String::fromUTF8("不修改音符") }, juce::String::fromUTF8("音符处理"));
    dialog->getComboBoxComponent("noteTiming")->setSelectedItemIndex(0,
        juce::dontSendNotification);
    dialog->addButton(juce::String::fromUTF8("确定"), 1);
    dialog->addButton(juce::String::fromUTF8("取消"), 0,
                      juce::KeyPress(juce::KeyPress::escapeKey));
    juce::Component::SafePointer<TimelineComponent> safe(this);
    dialog->enterModalState(true,
        juce::ModalCallbackFunction::create(
            [safe, dialog, quarterPosition](int result)
            {
                if (safe != nullptr && result == 1)
                {
                    const auto bpm = dialog->getTextEditorContents("bpm").getDoubleValue();
                    if (bpm >= 20.0 && bpm <= 400.0)
                        safe->model.setTempoChange(quarterPosition, bpm,
                            dialog->getComboBoxComponent("noteTiming")->getSelectedItemIndex() == 0);
                }
                delete dialog;
            }), false);
}

void TimelineComponent::mouseDown(const juce::MouseEvent& event)
{
    rememberPointer(event);
    grabKeyboardFocus();
    marqueePending = marqueeActive = false;
    if (event.y < rulerHeight && event.mods.isPopupMenu())
    {
        const auto seconds = std::max(0.0,
            static_cast<double>(event.position.x) / pixelsPerSecond);
        const auto barQuarters = static_cast<double>(std::max(1, snapshot.numerator))
            * 4.0 / static_cast<double>(std::max(1, snapshot.denominator));
        const auto quarterBar = std::max(1.0 / 256.0, barQuarters / 4.0);
        const auto clickedQuarter = snapshot.quarterPositionForSeconds(seconds);
        const auto snappedQuarter = std::max(0.0,
            std::floor((clickedQuarter + 1.0e-9) / quarterBar) * quarterBar);
        showTempoMenu(snappedQuarter, event.getScreenPosition());
        return;
    }
    if (event.y < rulerHeight)
    {
        if (onSeek) onSeek(std::max(0.0, static_cast<double>(event.x) / pixelsPerSecond));
        return;
    }
    if (event.mods.isAltDown() && !event.mods.isPopupMenu())
    { beginMarquee(event); return; }
    if (const auto hit = envelopeHit(event.position, false))
    {
        draggedClip.clear(); dragMode = DragMode::none;
        const auto* clip = findEnvelopeClip(hit->clipId);
        if (clip == nullptr) return;
        selectedEnvelopeClip = hit->clipId;
        selectedEnvelopeTime = clip->gainEnvelope[static_cast<std::size_t>(hit->pointIndex)].timeSeconds;
        if (event.mods.isPopupMenu()) { showEnvelopePointMenu(*hit,event.getScreenPosition()); return; }
        draggedEnvelopeClip = hit->clipId; draggedEnvelopeClipIndex = hit->trackIndex;
        draggedEnvelopeStart = clip->startSeconds; draggedEnvelopeDuration = clip->durationSeconds;
        draggedEnvelopePoint = hit->pointIndex;
        envelopeBeforeDrag = clip->gainEnvelope; envelopePreview = envelopeBeforeDrag;
        envelopeDragPosition = event.position;
        setMouseCursor(juce::MouseCursor::DraggingHandCursor); repaint(); return;
    }
    clearEnvelopeDrag(); selectedEnvelopeClip.clear();
    for (auto it = clipHits.rbegin(); it != clipHits.rend(); ++it)
        if (it->bounds.expanded(4.0f, 0.0f).contains(event.position))
        {
            const auto onGainKnob = gainControlBounds(it->bounds).contains(event.position);
            if (!event.mods.isPopupMenu() && !onGainKnob && (event.mods.isShiftDown() || event.mods.isCommandDown()))
            {
                auto ids = selectedClips;
                if (isClipSelected(it->id) && event.mods.isCommandDown()) std::erase(ids, it->id);
                else if (!isClipSelected(it->id)) ids.push_back(it->id);
                setSelectedClips(std::move(ids)); return;
            }
            if (!isClipSelected(it->id)) setSelectedClips({it->id});
            nativeFocusedNote.clear();selectedClip = it->id; notifySelection();
            if (event.mods.isPopupMenu())
            {
                draggedClip.clear();
                dragMode = DragMode::none;
                setMouseCursor(juce::MouseCursor::NormalCursor);
                // Capture the visible red playhead for both menu eligibility and the split action.
                if (onClipMenu) onClipMenu(selectedClip, playheadSeconds, event.getScreenPosition());
                repaint();
                return;
            }
            const auto muteBounds = juce::Rectangle<float>(it->bounds.getX() + 2.0f,
                it->bounds.getY() + 2.0f, 14.0f, 13.0f);
            if (muteBounds.contains(event.position))
            {
                model.setClipMuted(it->id, !it->muted);
                return;
            }
            draggedClip = it->id;
            draggedClipStart = it->startSeconds;
            draggedClipDuration = it->durationSeconds;
            draggedClipPreviewStart = draggedClipStart;
            draggedClipPreviewDuration = draggedClipDuration;
            draggedClipFadeIn = it->fadeInSeconds;
            draggedClipFadeOut = it->fadeOutSeconds;
            draggedClipPreviewFadeIn = draggedClipFadeIn;
            draggedClipPreviewFadeOut = draggedClipFadeOut;
            draggedAudioStart = it->audioStartSeconds;
            draggedAudioDuration = it->audioDurationSeconds;
            dragAnchorX = event.position.x;
            earliestDraggedStart = draggedClipStart;
            draggedClipTrack = juce::jlimit(0, static_cast<int>(snapshot.tracks.size())-1,
                static_cast<int>(std::floor((it->bounds.getCentreY()-rulerHeight)/rowHeight)));
            draggedTrackDelta = 0;
            firstDraggedTrack = lastDraggedTrack = draggedClipTrack;
            for (int row = 0; row < static_cast<int>(snapshot.tracks.size()); ++row)
                for (const auto& clip : snapshot.tracks[static_cast<std::size_t>(row)].clips)
                    if (isClipSelected(clip.id)) {
                        earliestDraggedStart = std::min(earliestDraggedStart, clip.startSeconds);
                        firstDraggedTrack = std::min(firstDraggedTrack, row);
                        lastDraggedTrack = std::max(lastDraggedTrack, row);
                    }
            const auto waveformTop = it->bounds.getY() + 19.0f;
            const auto fadeIn = juce::Point<float>(timeToX(it->audioStartSeconds + it->fadeInSeconds), waveformTop);
            const auto fadeOut = juce::Point<float>(timeToX(it->audioStartSeconds + it->audioDurationSeconds - it->fadeOutSeconds), waveformTop);
            const auto grip = std::min(14.0f, it->bounds.getWidth() * 0.25f);
            if (onGainKnob)
            {
                gainValueClick = gainControlLayout(it->bounds).value.contains(event.position);
                dragMode = DragMode::gain;
                draggedClipPreviewGain = it->gain;
                draggedClipGainDb = juce::Decibels::gainToDecibels(it->gain, -60.0f);
                gainDragY = event.position.y;
                setTooltip({});
                setMouseCursor(juce::MouseCursor::UpDownResizeCursor);
            }
            else if (event.position.getDistanceFrom(fadeIn) <= 5.0f)
            {
                dragMode = DragMode::fadeIn;
                setMouseCursor(juce::MouseCursor::LeftRightResizeCursor);
            }
            else if (event.position.getDistanceFrom(fadeOut) <= 5.0f)
            {
                dragMode = DragMode::fadeOut;
                setMouseCursor(juce::MouseCursor::LeftRightResizeCursor);
            }
            else if (event.position.x <= it->bounds.getX() + grip)
            {
                dragMode = DragMode::resizeLeft;
                setMouseCursor(juce::MouseCursor::LeftRightResizeCursor);
            }
            else if (event.position.x >= it->bounds.getRight() - grip)
            {
                dragMode = DragMode::resizeRight;
                setMouseCursor(juce::MouseCursor::LeftRightResizeCursor);
            }
            else
            {
                dragMode = DragMode::move;
                setMouseCursor(juce::MouseCursor::DraggingHandCursor);
            }
            repaint();
            return;
        }
    if (event.mods.isPopupMenu())
    {
        // Below the ruler and on no clip: the space around the tracks.  It
        // used to move the playhead, which no other right-click in the app
        // does, and which left no way to act on the lane itself.
        if (onEmptyAreaMenu) onEmptyAreaMenu(event.getScreenPosition());
        return;
    }
    beginMarquee(event);
}

void TimelineComponent::mouseDrag(const juce::MouseEvent& event)
{
    if(event.getDistanceFromDragStart()>=5)gainValueClick=false;
    if (draggedEnvelopeClip.isNotEmpty())
    {
        auto& point = envelopePreview[static_cast<std::size_t>(draggedEnvelopePoint)];
        const auto low = std::max(0.0,draggedEnvelopePoint > 0 ? envelopePreview[static_cast<std::size_t>(draggedEnvelopePoint-1)].timeSeconds+0.001 : 0.0);
        const auto high = std::min(draggedEnvelopeDuration,static_cast<std::size_t>(draggedEnvelopePoint+1) < envelopePreview.size()
            ? envelopePreview[static_cast<std::size_t>(draggedEnvelopePoint+1)].timeSeconds-0.001 : draggedEnvelopeDuration);
        if (high >= low)
            point.timeSeconds = juce::jlimit(low,high,point.timeSeconds
                + static_cast<double>(event.position.x-envelopeDragPosition.x)/pixelsPerSecond);
        const auto height = envelopeArea(draggedEnvelopeClipIndex).getHeight()*0.5f;
        const auto delta = (envelopeDragPosition.y-event.position.y)/std::max(1.0f,height);
        point.gainDb = trackGainDbFromLevel(trackGainLevelFromDb(point.gainDb)+delta*(event.mods.isShiftDown()?0.1f:1.0f));
        envelopeDragPosition = event.position;
        repaint(); return;
    }
    if (marqueePending)
    {
        if (event.position.getDistanceFrom(marqueeAnchor) >= 4.0f) marqueeActive = true;
        if (marqueeActive)
        {
            updateMarquee(event.position);
            if (auto* viewport = findParentComponentOfClass<juce::Viewport>())
            {
                const auto p = viewport->getLocalPoint(this, event.getPosition());
                viewport->autoScroll(p.x, p.y, 18, 18);
            }
        }
        return;
    }
    if (draggedClip.isEmpty()) return;
    const auto snap = [&](double seconds) { return snapToGrid(seconds); };
    const auto delta = static_cast<double>(event.position.x - dragAnchorX) / pixelsPerSecond;
    if (dragMode == DragMode::gain)
    {
        const auto deltaY = gainDragY - event.position.y;
        gainDragY = event.position.y;
        if (deltaY != 0.0f)
        {
            draggedClipGainDb = juce::jlimit(-60.0f, 12.0f,
                draggedClipGainDb + deltaY * (event.mods.isShiftDown() ? 0.025f : 0.25f));
            draggedClipPreviewGain = juce::Decibels::decibelsToGain(draggedClipGainDb, -60.0f);
        }
    }
    else if (dragMode == DragMode::fadeIn)
        draggedClipPreviewFadeIn = juce::jlimit(0.0, draggedAudioDuration,
            static_cast<double>(event.position.x) / pixelsPerSecond - draggedAudioStart);
    else if (dragMode == DragMode::fadeOut)
        draggedClipPreviewFadeOut = juce::jlimit(0.0, draggedAudioDuration,
            draggedAudioStart + draggedAudioDuration
                - static_cast<double>(event.position.x) / pixelsPerSecond);
    else if (dragMode == DragMode::resizeLeft)
    {
        const auto end = draggedClipStart + draggedClipDuration;
        draggedClipPreviewStart = juce::jlimit(0.0, end - 0.01,
                                                snap(draggedClipStart + delta));
        draggedClipPreviewDuration = end - draggedClipPreviewStart;
    }
    else if (dragMode == DragMode::resizeRight)
    {
        const auto end = std::max(draggedClipStart + 0.01,
                                  snap(draggedClipStart + draggedClipDuration + delta));
        draggedClipPreviewDuration = end - draggedClipStart;
    }
    else if (dragMode == DragMode::move) {
        draggedClipPreviewStart = draggedClipStart + std::max(-earliestDraggedStart,
            snap(draggedClipStart + delta) - draggedClipStart);
        const auto pointedRow = static_cast<int>(std::floor((event.position.y-rulerHeight)/rowHeight));
        draggedTrackDelta = juce::jlimit(-firstDraggedTrack,
            static_cast<int>(snapshot.tracks.size())-1-lastDraggedTrack, pointedRow-draggedClipTrack);
        draggedClipPreviewStart=draggedClipStart+constrainedNativeClipMoves(snapshot,selectedClips,
            draggedClipPreviewStart-draggedClipStart,draggedTrackDelta);
        if (auto* viewport = findParentComponentOfClass<juce::Viewport>()) {
            const auto p = viewport->getLocalPoint(this, event.getPosition());
            viewport->autoScroll(p.x, p.y, 18, 18);
        }
    }
    repaint();
}

void TimelineComponent::mouseUp(const juce::MouseEvent& event)
{
    if(dragMode==DragMode::gain && gainValueClick && event.getDistanceFromDragStart()<5)
    {
        const auto clipId=draggedClip;
        draggedClip.clear();dragMode=DragMode::none;gainValueClick=false;
        setMouseCursor(juce::MouseCursor::NormalCursor);repaint();
        if(onClipGainRequested)onClipGainRequested(clipId);
        return;
    }
    gainValueClick=false;
    if (draggedEnvelopeClip.isNotEmpty())
    {
        selectedEnvelopeTime = envelopePreview[static_cast<std::size_t>(draggedEnvelopePoint)].timeSeconds;
        model.setClipGainEnvelope(draggedEnvelopeClip,envelopePreview);
        clearEnvelopeDrag(); repaint(); return;
    }
    if (marqueePending)
    {
        if (!marqueeActive && !marqueeAdditive && onSeek)
            onSeek(std::max(0.0, static_cast<double>(marqueeAnchor.x) / pixelsPerSecond));
        marqueePending = marqueeActive = false;
        repaint(); return;
    }
    if (draggedClip.isNotEmpty())
    {
        if (dragMode == DragMode::gain)
            model.setClipGain(draggedClip, draggedClipPreviewGain);
        else if (dragMode == DragMode::fadeIn || dragMode == DragMode::fadeOut)
            model.setClipFades(draggedClip, draggedClipPreviewFadeIn,
                               draggedClipPreviewFadeOut);
        else if (dragMode == DragMode::resizeLeft || dragMode == DragMode::resizeRight)
            model.trimClip(draggedClip, draggedClipPreviewStart,
                             draggedClipPreviewDuration);
        else if (dragMode == DragMode::move
            && (draggedTrackDelta != 0 || std::abs(draggedClipPreviewStart - draggedClipStart) > 1.0e-9))
            model.moveClips(selectedClips, draggedClipPreviewStart - draggedClipStart, draggedTrackDelta);
    }
    draggedClip.clear();
    draggedTrackDelta = 0;
    dragMode = DragMode::none;
    setMouseCursor(juce::MouseCursor::NormalCursor);
    repaint();
}
}
#include "TimelineGainEnvelope.h"
