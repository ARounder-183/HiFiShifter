#pragma once

#include "I18n.h"
#include "AudioEngine.h"
#include "ProjectModel.h"
#include "NativeSharedEnvelope.h"
#include "NativePitchVoicingDisplay.h"
#include "backend/UtauRenderer.h"
#include "SampleSettings.h"
#include "NoteDanceAnimation.h"
#include "OtoRegionGuides.h"
#include <juce_audio_utils/juce_audio_utils.h>
#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace hachi
{
class NativePitchVoicingCache;
class PianoRollComponent final : public juce::Component,
                                 public juce::SettableTooltipClient,
                                 private juce::ChangeListener,
                                 private juce::Timer
{
public:
    // Puts everything that decides where a note's consonant sits back to what
    // the voicebank says.
    void resetConsonant(const juce::String& noteId);
    // Deletes the selection and the time it took up, closing the gap.
    void deleteSelectedNotesRippling(const juce::String& noteId);
    enum class Tool { note, draw, line, points, amplitude, flagCurve, connect, trim };
    // An explicit view/tool choice, never inferred from a track's sound engine.
    enum class DrawAction { pitchCurve, notes };
    enum class PitchScope { focusedClip, currentTrack };

    PianoRollComponent(ProjectModel& modelToUse, const I18n& stringsToUse);
    void setNoteDanceEnabled(bool enabled)
    {
        if (noteDance.isEnabled() == enabled) return;
        noteDance.setEnabled(enabled);
        repaint();
    }
    bool isNoteDanceEnabled() const { return noteDance.isEnabled(); }
    void advanceNoteDance()
    {
        if (!noteDance.isEnabled() || !isShowing()) return;
        noteDance.advance();
        if (const auto* view = findParentComponentOfClass<juce::Viewport>())
            repaint(view->getViewArea());
        else
            repaint();
    }
    // Offline read-outs for --smoke-piano-roll.  The roll is only reachable
    // through a window otherwise, which left its geometry unverifiable.
    struct DiagnosticNote
    {
        juce::String id, label;
        double start = 0.0, end = 0.0;
        double soundingStart = 0.0, soundingEnd = 0.0;
        bool utau = false;
        std::vector<AmplitudeEnvelopePoint> envelope;
    };
    [[nodiscard]] std::vector<DiagnosticNote> diagnosticNotes() const;
    // Whether the roll gave this note a sounding stretch at all.  A rest
    // has none: diagnosticNotes falls back to the written bounds, which
    // says nothing about whether anything sounds there.
    [[nodiscard]] bool diagnosticHasSoundingSpan(const juce::String& id) const
    {
        return utauSoundSpans.count(id.toStdString()) > 0;
    }
    // The stretch the roll believes a note sounds over.  What is drawn and
    // what is mixed have to say the same thing about where a note stops, so a
    // check reads both.
    [[nodiscard]] std::pair<double, double> diagnosticSoundingSpan(
        const juce::String& id) const
    {
        const auto found = utauSoundSpans.find(id.toStdString());
        return found != utauSoundSpans.end() ? found->second
                                             : std::pair<double, double> { 0.0, 0.0 };
    }
    // The envelope as the roll draws it, in seconds from the note's start.
    // The renderer shapes the note with its own copy of this; the two have to
    // end in the same place or the sound stops somewhere the drawing does not.
    [[nodiscard]] std::vector<AmplitudeEnvelopePoint> diagnosticDrawnEnvelope(
        const juce::String& id) const;
    // Model changes arrive through an async broadcast, which never lands
    // without a message loop; the offline check reads the roll straight
    // after editing it, so it needs the snapshot brought up to date now.
    void diagnosticRefresh();
    [[nodiscard]] juce::Rectangle<float> diagnosticOverlapSourceBounds(const juce::String& id) const
    { for(const auto& hit:nativeOverlapSourceHits)if(hit.noteId==id)return hit.bounds;return {}; }
    void setShowOverlap(bool show) { if(showOverlap==show)return;showOverlap=show;nativeOverlapLines.clear();repaint(); }
    [[nodiscard]] bool showsOverlap() const { return showOverlap; }
    [[nodiscard]] const std::vector<std::pair<juce::String,juce::Rectangle<float>>>& diagnosticOverlapLines() const { return nativeOverlapLines; }
    [[nodiscard]] std::size_t diagnosticOverlapCardCount() const { return nativeOverlapCardCount; }
    static constexpr int materialSelectorNoteMenuItem = 32;
    [[nodiscard]] juce::String diagnosticOverlapSelectorTarget() const
    { return nativeOverlapSelector ? nativeOverlapSelector->targetNoteId : juce::String{}; }
    [[nodiscard]] juce::Rectangle<float> diagnosticOverlapSelectorCloseBounds() const { return nativeOverlapSelectorClose; }
    // Update immediately before focusing notes created by a synchronous edit.
    void refreshFromModel();
    // Builds the same sounding spans/curves as the loudness lane, then commits once.
    int crossfadeSelectedNotes(bool curved = false);
    ~PianoRollComponent() override;

    void setPixelsPerSecond(float value);
    void setRowHeight(float value);
    void setSourceEditMode(bool enabled);
    void cancelPendingEditingForTrackChange();
    void setFocusedClip(const juce::String& clipId);
    void setFocusedTrack(const juce::String& trackId);
    void setShowNoteLabels(bool enabled);
    void setShowPhonemeInfo(bool enabled) { showPhonemeInfo=enabled; repaint(); }
    bool phonemeInformationVisible(const juce::String& id) const
    { return showPhonemeInfo || selectedNotes.contains(id.toStdString()) || selectedNote==id; }
    std::vector<backend::UtauPhonemeSpan> diagnosticPhonemeSpans(const juce::String& id) const
    { const auto it=diffSingerPhonemes.find(id.toStdString());return it==diffSingerPhonemes.end()?std::vector<backend::UtauPhonemeSpan>{}:it->second; }
    // Read a voicebank on a background thread rather than stopping the window
    // for it.  Until a bank is read its notes are drawn where they were last
    // drawn -- or, the first time, at their written place -- and the layout is
    // redone once it is in.  Off unless asked for, so a check that builds a
    // roll gets its lead-ins at once.
    void setReadsVoicebankInBackground(bool enabled) { readsVoicebankInBackground = enabled; }
    void setShowWaveforms(bool enabled);
    // The rendered UTAU notes, as peaks.  A UTAU note has no source file to
    // take a thumbnail of, so what is drawn is what came back from the
    // resampler -- and only for a note that still hashes to what produced it.
    void setUtauNoteWaveforms(
        std::shared_ptr<const std::vector<UtauNoteWaveform>> waveforms);
    // Its own switch, separate from the source-audio waveforms: off means off,
    // whatever has been rendered.
    void setShowNativeRenderedWaveforms(bool show);
    void setNativeClipWaveforms(std::shared_ptr<const std::vector<NativeRenderedWaveform>> waveforms);
    void setShowNativeWaveforms(bool show) { showNativeWaveforms = show; repaint(); }
    void setShowLyrics(bool show) { showLyrics = show; repaint(); }
    void setShowUtauWaveforms(bool enabled);

    // A line of lyrics, one word per note.  Whitespace separates them, and
    // that includes the full-width space a Chinese or Japanese keyboard
    // produces -- typing the line in the input method that wrote the lyrics
    // must not silently put the whole line on one note.  Pure, and public so
    // a check reads the same splitting the dialog does.
    [[nodiscard]] static juce::StringArray splitBatchLyrics(const juce::String& text);

    // What the note menu offers, per mode.
    //
    // Most of it only means something on a UTAU track.  Timing and the region
    // editor come from a voicebank entry; flags, the four regions and the
    // forced consonant reset are voicebank ideas; a lyric names the sample to
    // sing.  Pitch anchors and vibrato are shared editing concepts, so they
    // remain available on ordinary audio/Melodyne-compatible tracks as well.
    //
    // Returns the item ids in menu order.  One list for both menus, so they
    // cannot drift apart over what exists and a new item cannot be added to
    // one and forgotten in the other.  Pure, and public so a check reads the
    // same list the menu is built from.
    [[nodiscard]] static std::vector<int> noteMenuItemsFor(bool utauTrack);
    // The pitch line laid flat on the note's own pitch, with nothing left but
    // an anchor at each end.  On ordinary audio tracks this resets the target
    // pitch line without changing the analysed source contour; on UTAU tracks
    // it is the same anchor baseline the renderer follows.
    void flattenPitchLine(const juce::String& noteId, bool snapNativeToSemitone = false);
    void restoreOriginalPitch(const juce::String& noteId);
    // A frequency as a MIDI pitch, and back.  Twelve-tone equal temperament
    // against concert A4 = 440 Hz, which is what the roll's vertical axis is.
    //
    // Nothing comes back for a frequency that names no pitch: zero and below
    // are not pitches at all, and outside the MIDI range there is no row for
    // an anchor to sit on -- a stray keystroke would otherwise throw the point
    // somewhere the line cannot be seen or dragged back from.
    //
    // Pure, and public so a check reads the same rule the dialog applies.
    [[nodiscard]] static std::optional<float> anchorMidiForFrequency(double hertz);
    [[nodiscard]] static double anchorFrequencyForMidi(float midi);
    // The anchors as they are after one of them is given a frequency.  Empty
    // when the frequency names no pitch or the index names no point: nothing
    // is written at all then, rather than a point being moved somewhere it
    // cannot be seen.  A curve always has at least two points, so an empty
    // result cannot be mistaken for one.
    //
    // Pure, and public so a check can follow what the box does without one.
    [[nodiscard]] static std::vector<PitchCurveEditPoint> anchorsWithFrequency(
        std::vector<PitchCurveEditPoint> anchors, int anchorIndex, double hertz);
    // The anchor menu as it is actually built, for a check to walk.  The one
    // that opens on screen is this menu; a list of what ought to be in it
    // could agree with the rules and disagree with the menu.
    // Whether the anchor menu's handler acts on a choice.  It used to be a
    // pair of literals sitting beside the handler, so an item added to the
    // menu and not to them appeared and did nothing.  Pure, and public so a
    // check can ask it of every id the menu actually offers.
    [[nodiscard]] static bool anchorMenuChoiceHandled(int choice);
    [[nodiscard]] std::vector<int> diagnosticAnchorMenuIds(
        const juce::String& noteId, int anchorIndex);
    [[nodiscard]] std::vector<int> diagnosticEnabledAnchorMenuIds(
        const juce::String& noteId, int anchorIndex);

    // Where a paste should land: the grid line nearest the pointer, or nothing
    // when the pointer is not over the roll.
    //
    // Without it a paste went to the start of the selection, which after a
    // copy is the very notes that were copied -- and off a UTAU track notes
    // are placed rather than pushed along, so the copy landed exactly on top
    // of its original and nothing appeared to happen.
    [[nodiscard]] std::optional<double> pasteAnchorSeconds() const;
    // The ids the built menu really carries, in order.
    // The same, less the items offered greyed: whether a thing can be
    // reached is a separate question from whether it is listed.
    [[nodiscard]] std::vector<int> diagnosticEnabledNoteMenuIds(
        const juce::String& noteId) const;
    // What choosing an item on the note menu does.  Public, because inside
    // the popup's own callback nothing could ask whether an item does
    // anything at all.
    void applyNoteMenuChoice(const juce::String& noteId, int result, double split,
                             const std::vector<juce::String>& mergeIds,
                             double contextSeconds = std::numeric_limits<double>::quiet_NaN());
    [[nodiscard]] std::vector<int> diagnosticNoteMenuIds(const juce::String& noteId) const;
    // Both of those include what is inside 高级.  This is the menu row by row
    // instead: an item's id, a submenu as its name with its ids in brackets,
    // a separator as "-".
    [[nodiscard]] juce::String diagnosticNoteMenuLabel(const juce::String& noteId, int itemId) const;
    [[nodiscard]] std::vector<juce::String> diagnosticNoteMenuLayout(
        const juce::String& noteId) const;

    // The steps a drawn note may grow by: a 128th of a bar up to a quarter
    // of one, doubling.  Generated rather than written out, so doubling is
    // the rule rather than a list that could be typed wrong.  Coarsest last,
    // which is the order they are offered in.
    [[nodiscard]] static std::vector<int> drawLengthDivisions();
    // Ignores a value that is not one of those -- a settings file can say
    // anything, and a step of "1/37 of a bar" is not a thing to honour.
    void setDrawLengthDivision(int division);
    [[nodiscard]] int drawLengthDivision() const { return drawLengthDivisionValue; }

    // How long a note being drawn is, given how far the pointer has travelled
    // from where it went down.  Whole units of a 64th of a bar, at least one,
    // and never past what is free.  Pure, and public so a check reads the same
    // arithmetic the drag does.
    [[nodiscard]] static double drawnLengthFor(double draggedSeconds,
                                               double unitSeconds,
                                               double availableSeconds);
    // Is a note being drawn, and where it currently reaches -- for a check to
    // read without painting.
    [[nodiscard]] bool diagnosticDrawingNote() const { return dragMode == DragMode::drawNewNote; }
    [[nodiscard]] double diagnosticDrawnStart() const { return drawStartSeconds; }
    [[nodiscard]] double diagnosticDrawnLength() const { return drawLengthSeconds; }

    [[nodiscard]] bool showsUtauWaveforms() const { return showUtauWaveforms; }
    // Where a note's waveform is drawn, in seconds on the timeline: the same
    // span the paint uses, so a check reads what is on screen rather than a
    // second copy of the rule.
    [[nodiscard]] std::optional<std::pair<double, double>> diagnosticWaveformSpan(
        const juce::String& noteId) const;
    // The orange outline around what a note actually sounds, and the
    // envelope shape inside it.  Independent: either alone still leaves a
    // note visible, since the nominal baseline is always drawn.
    void setShowNoteRange(bool enabled);
    void setShowEnvelope(bool enabled);
    // The line the note is actually sung along -- the anchor curve in the
    // UTAU modes, the analysed and edited contours elsewhere.  The anchor
    // dots stay drawn when it is off, so the point tool can still be used to
    // put the line back where it belongs.
    void setShowPitchLine(bool enabled);
    [[nodiscard]] bool showsPitchLine() const { return showPitchLine; }
    void setShowOriginalPitchLine(bool show) { showOriginalPitchLine = show; repaint(); }
    [[nodiscard]] bool showsOriginalPitchLine() const { return showOriginalPitchLine; }
    void setConsonantPitchDashed(bool show);
    [[nodiscard]] bool showsConsonantPitchDashed() const { return consonantPitchDashed; }
    [[nodiscard]] bool diagnosticConsonantAnalysisPending() const;
    [[nodiscard]] juce::Path diagnosticDiffSingerPitchReference(const juce::String& noteId) const;
    // Test seam: lets a harness paint the same strip with and without the
    // off-screen skip, which is the only way to tell a note that was correctly
    // skipped from one that was wrongly dropped.
    void diagnosticSetCulling(bool enabled) { cullOffscreenNotes = enabled; }
    // Whether the lyric box is showing, for a check to read without looking
    // at pixels.
    [[nodiscard]] bool diagnosticAliasEditorOpen() const
    {
        return inlineAliasNoteId.isNotEmpty();
    }
    // Closes it without writing, so a check can go on to the next case.
    void diagnosticCancelAliasEdit() { finishInlineAliasEdit(false); }
    // Which note the box is on and what it holds, for the Tab check.
    [[nodiscard]] juce::String diagnosticAliasEditorNoteId() const
    {
        return inlineAliasNoteId;
    }
    [[nodiscard]] juce::String diagnosticAliasEditorText() const
    {
        return inlineAliasEditor.getText();
    }
    [[nodiscard]] bool diagnosticAliasEditorHasFocus() const
    {
        return inlineAliasEditor.hasKeyboardFocus(false);
    }
    // The whole lyric highlighted, so the first keystroke replaces it.
    [[nodiscard]] bool diagnosticAliasEditorAllSelected() const
    {
        return inlineAliasEditor.getHighlightedRegion().getLength()
            == inlineAliasEditor.getTotalNumChars();
    }
    [[nodiscard]] juce::Rectangle<int> diagnosticAliasEditorBounds() const
    {
        return inlineAliasEditor.getBounds();
    }
    // Puts text in the box without accepting it, as typing would.
    void diagnosticTypeInAliasEditor(const juce::String& text)
    {
        inlineAliasEditor.setText(text, false);
    }
    // Whether the point tool has taken hold of a pitch anchor.
    // Which note's lead-in handle a press at this point takes hold of, or
    // empty.  Where two handles overlap only one of them can be had, and the
    // note behind a 拼字 note was the one that never could.
    [[nodiscard]] juce::String diagnosticConsonantHandleAt(
        juce::Point<float> position) const
    {
        const auto handle = consonantHandleAt(position);
        return handle ? handle->noteId : juce::String();
    }
    // The pitch handles a note is drawn and edited with.
    // Where a vibrato handle of a note sits on screen -- "length" (its start),
    // "end", "fadeIn", "fadeOut", "depth", "cycle" or "offset".
    [[nodiscard]] std::optional<juce::Point<float>> diagnosticVibratoHandle(
        const juce::String& noteId, const juce::String& which) const;
    [[nodiscard]] std::vector<PitchCurveEditPoint> diagnosticPitchAnchors(const juce::String& id)
    {
        if (const auto* note = findNote(id)) return pitchAnchorsFor(*note);
        return {};
    }
    // The pitch line drawn for a note, at an absolute time, and the stretch it
    // is drawn over -- read the way paint reads them.
    // The automatic transition between two adjacent notes: from where this
    // note's own line ends to where the next one's begins.
    struct TransitionBridge
    {
        double startSeconds = 0.0;   // absolute
        double endSeconds = 0.0;
        float startMidi = 60.0f;
        float endMidi = 60.0f;
    };
    [[nodiscard]] std::optional<TransitionBridge> diagnosticTransitionBridge(
        const juce::String& id);
    [[nodiscard]] std::optional<float> diagnosticPitchLineAt(const juce::String& id,
                                                             double absoluteSeconds);
    [[nodiscard]] std::optional<std::pair<double, double>> diagnosticPitchLineSpan(
        const juce::String& id);
    // The handles a note offers: on a shared line, only those in the stretch
    // its own points decide.
    [[nodiscard]] std::vector<PitchCurveEditPoint> diagnosticOfferedPitchAnchors(
        const juce::String& id);
    [[nodiscard]] bool diagnosticDraggingAnchor() const
    {
        return dragMode == DragMode::pointPitch;
    }
    // Types a lyric and accepts it, which is the step that tells the window a
    // note has been named -- and the window answers by converting the track.
    void diagnosticCommitAliasEdit(const juce::String& text)
    {
        inlineAliasEditor.setText(text, false);
        finishInlineAliasEdit(true);
    }
    // The middle of a pitch row, so a harness can aim a click at one.  A
    // check that clicks the wrong row would pass for the wrong reason, so the
    // notes it makes are checked for their pitch as well.
    [[nodiscard]] float diagnosticYForMidi(float midi) const
    {
        return midiToY(midi) + rowHeight * 0.5f;
    }
    [[nodiscard]] std::size_t diagnosticHitCount() const { return noteHits.size(); }
    [[nodiscard]] float diagnosticHitX(std::size_t index) const
    {
        return index < noteHits.size() ? noteHits[index].bounds.getX() : 0.0f;
    }
    [[nodiscard]] juce::Rectangle<float> diagnosticHitBounds(std::size_t index) const
    {
        return index < noteHits.size() ? noteHits[index].bounds
                                       : juce::Rectangle<float>();
    }
    // Test seam: puts the roll into the state a note drag would, so a harness
    // can ask what a drag would repaint and check that it is enough.
    void diagnosticBeginMoveDrag(const juce::String& noteId, double deltaSeconds);
    [[nodiscard]] juce::Rectangle<int> diagnosticDragRepaintArea(float mouseX) const;
    // Where the roll draws the three region boundaries of a note, in
    // seconds, exactly as the paint places them.
    [[nodiscard]] std::array<double, 3> diagnosticRegionEdges(std::size_t index) const;
    // How many regions the roll is drawing for this note.
    [[nodiscard]] int diagnosticRegionCount(std::size_t index) const;
    // The per-region flag boxes this note would be offered, as "label=value".
    [[nodiscard]] juce::StringArray diagnosticRegionFlagFields(
        const juce::String& noteId) const;
    // The two amounts the gap items insert, at this note's own tempo.
    [[nodiscard]] bool diagnosticGapItemsEnabled(const juce::String& noteId) const
    {
        return isUtauNote(noteId);
    }
    // The gap the roll would find under this point, as
    // "previous..next from..to", or empty when there is none.
    [[nodiscard]] juce::String diagnosticFlagPointValueText(
        const juce::String& noteId, int index) const
    {
        return flagPointValueText(noteId, index);
    }
    [[nodiscard]] juce::String diagnosticFlagPointValueBlurb(
        const juce::String& noteId, int index) const
    {
        return flagPointValueBlurb(noteId, index);
    }
    [[nodiscard]] juce::String diagnosticGapAt(juce::Point<float> position) const
    {
        const auto gap = gapAt(position);
        if (!gap) return {};
        return gap->previousId + ".." + gap->nextId + " "
            + juce::String(gap->fromSeconds, 4) + ".."
            + juce::String(gap->toSeconds, 4);
    }
    // Drives what the two gap items do, without the menu.
    void diagnosticCloseGapAt(juce::Point<float> position)
    {
        if (const auto gap = gapAt(position))
            model.closeGapBeforeNote(gap->nextId, gap->toSeconds - gap->fromSeconds);
    }
    juce::String diagnosticInsertNoteAt(juce::Point<float> position)
    {
        const auto gap = gapAt(position);
        return gap ? insertNoteIntoGap(*gap) : juce::String();
    }
    [[nodiscard]] double diagnosticGapUnitSeconds(const juce::String& noteId) const
    {
        return gapUnitSecondsAt(std::max(0.0, absoluteStartOf(noteId)));
    }
    [[nodiscard]] double diagnosticQuarterBarSeconds(const juce::String& noteId) const
    {
        return quarterBarSecondsAt(std::max(0.0, absoluteStartOf(noteId)));
    }
    // Whether a region boundary can be grabbed at this point, and where to aim.
    [[nodiscard]] bool diagnosticJieHandleAt(juce::Point<float> position) const
    {
        return jieSplitHandleAt(position).has_value();
    }
    [[nodiscard]] float diagnosticEdgeX(double seconds) const
    {
        return timeToX(seconds);
    }
    [[nodiscard]] float diagnosticNoteY(std::size_t index) const;
    // Test seam: puts the roll into the state a consonant drag would, so the
    // boundaries can be asked about mid-drag, which is where they went wrong.
    void diagnosticBeginConsonantDrag(const juce::String& noteId,
                                      double previewPreutterance);
    // Test seam: takes hold of the note-front handle through the same lookup a
    // press goes through, so what the handle offers and what a release writes
    // can both be asked about.  False when the note has no such handle.
    bool diagnosticGrabConsonantHandle(std::size_t index);
    [[nodiscard]] double diagnosticConsonantRangeMin() const
    {
        return consonantMinimumPreutterance;
    }
    [[nodiscard]] double diagnosticConsonantRangeMax() const
    {
        return consonantMaximumPreutterance;
    }
    [[nodiscard]] bool diagnosticConsonantSetsPin() const { return consonantSetsPin; }
    [[nodiscard]] double diagnosticConsonantPreview() const { return previewConsonantPreutterance; }
    [[nodiscard]] int diagnosticConsonantVelocity() const { return previewConsonantVelocity; }
    // Drives the same arithmetic a drag does, from the lead-in the cursor
    // would be asking for.
    void diagnosticDragConsonantTo(double preutteranceSeconds)
    {
        // Asked for a lead-in outright, which is a drag and not a click.
        consonantDragTravelled = true;
        dragConsonantTo(consonantNoteAbsoluteStart - preutteranceSeconds);
    }
    void diagnosticReleaseDrag() { finishDrag(); }
    void diagnosticBeginResizeDrag(const juce::String& noteId,
                                   double previewStart, double previewDuration);
    // The strip the playhead asked to have repainted when it last moved.
    [[nodiscard]] juce::Rectangle<int> diagnosticPlayheadBand() const
    {
        return playheadBand;
    }
    [[nodiscard]] double diagnosticPreviewMoveDelta() const
    {
        return previewMoveDeltaSeconds;
    }
    // Gives a note that has just been given a lyric the shape it is already
    // being drawn with, so it sounds like its neighbours straight away.
    void ensureDefaultEnvelope(const juce::String& noteId);
    int applyTailFade(int mode);
    [[nodiscard]] std::vector<OtoRegionGuide> otoRegionGuidesFor(const juce::String& noteId) const;
    [[nodiscard]] std::vector<AmplitudeEnvelopePoint> tailFadeBaseEnvelope(const juce::String& id) const;
    [[nodiscard]] std::optional<backend::UtauTailFadeSpan> headEnvelopeSpanFor(const juce::String& noteId) const;
    [[nodiscard]] std::optional<backend::UtauTailFadeSpan> tailFadeSpanFor(const juce::String& noteId) const;
    [[nodiscard]] std::optional<backend::UtauTailFadeSpan> diagnosticTailFadeSpan(const juce::String& noteId) const {return tailFadeSpanFor(noteId);}
    [[nodiscard]] std::vector<AmplitudeEnvelopePoint> diagnosticTailFadePicture(const juce::String& noteId) const;

    // A loudness envelope preset, point by point.  Each point is placed by
    // what a note is laid out by rather than by clock time -- after the moment
    // it starts sounding, after its beat, or before the moment it stops -- so
    // a rise can reach full level a set time into the vowel whatever the
    // consonant in front of it is.  A note starts and ends in silence; the
    // points are what lies between.
    struct EnvelopePresetPoint
    {
        enum class From { soundStart, beat, soundEnd };
        From from = From::soundStart;
        double seconds = 0.0;
        float gainDb = 0.0f;
        // The stretch from this point to the next runs straight in amplitude.
        bool linearToNext = false;
    };
    struct EnvelopePreset
    {
        juce::String name;
        juce::String tip;
        std::vector<EnvelopePresetPoint> points;
        // The stretch from the silent start to the first point.
        bool linearRise = false;
    };
    // The toolbar's presets, in the order they are offered.
    [[nodiscard]] static const std::vector<EnvelopePreset>& envelopePresets();
    // The envelope a preset gives a note sounding from `first` to `last`,
    // both in seconds from its beat.
    [[nodiscard]] static std::vector<AmplitudeEnvelopePoint> envelopePresetPoints(
        const EnvelopePreset& preset, double first, double last);
    // Writes a preset over each selected note, sized to what that note
    // actually sounds.
    int applyEnvelopePreset(const EnvelopePreset& preset);
    // The same for a two-ramp shape given by its numbers, straight in dB:
    // attack, release, and the level the hold ends on.
    int applyEnvelopePreset(double attackSeconds, double releaseSeconds,
                            float plateauEndDb);
    // An envelope with a point added at this time, on the line where it
    // already runs.  What a double-click in the envelope lane does.
    [[nodiscard]] static std::vector<AmplitudeEnvelopePoint> envelopeWithPointAt(
        std::vector<AmplitudeEnvelopePoint> envelope, double timeSeconds);
    // Test seams: the level the roll reads between two points, and the shape
    // an envelope takes when it is carried to another note.
    // Where the loudness lane puts a level, so a check can aim at a handle.
    [[nodiscard]] float diagnosticAmplitudeLaneY(float gainDb) const
    {
        return amplitudeLaneY(gainDb);
    }
    [[nodiscard]] std::vector<juce::Point<float>> diagnosticCvvcEnvelopeHandles(
        const juce::String& noteId, int part, bool amplitude) const;
    [[nodiscard]] juce::Rectangle<float> diagnosticCvvcPhonemeTab(const juce::String&, int part) const;
    [[nodiscard]] juce::Rectangle<float> diagnosticCvvcBoundary(const juce::String&, int part) const;
    [[nodiscard]] bool hasSelectedCvvcPhoneme() const;
    struct CvvcEnvelopeSelection
    {
        juce::String noteId, label;
        int part = 0;
        double from = 0, to = 0;
        backend::CvvcEnvelopeCurve points;
    };
    [[nodiscard]] std::optional<CvvcEnvelopeSelection> selectedCvvcEnvelope() const;
    // A note's envelope as the lane shows it -- the shape with its base value
    // already in it, which is what the audio will be shaped by.
    [[nodiscard]] std::vector<AmplitudeEnvelopePoint> diagnosticDisplayEnvelope(
        const juce::String& noteId) const
    {
        for (const auto& track : snapshot.tracks)
            for (const auto& clip : track.clips)
                for (const auto& note : clip.notes)
                    if (note.id == noteId)
                        return displayAmplitudeEnvelope(note,
                            clip.startSeconds + note.startSeconds);
        return {};
    }
    [[nodiscard]] static float diagnosticAmplitudeDbAt(
        const std::vector<AmplitudeEnvelopePoint>& points, double timeSeconds)
    {
        return amplitudeDbAt(points, timeSeconds);
    }
    // What the paint draws for a note, bucket by bucket: the picture in its
    // row and the faint layer behind the envelope lane, each as a level.
    // Empty when the note has no waveform the roll would draw.
    struct DrawnNoteWaveform
    {
        double leadInSeconds = 0.0;
        std::vector<float> picture;
        std::vector<float> ghost;
    };
    [[nodiscard]] std::optional<DrawnNoteWaveform> diagnosticDrawnNoteWaveform(
        const juce::String& noteId) const;
    [[nodiscard]] std::vector<AmplitudeEnvelopePoint> diagnosticMapAmplitudeEnvelope(
        const std::vector<AmplitudeEnvelopePoint>& source,
        const juce::String& sourceNoteId, const juce::String& targetNoteId) const;
    void setSampleRegions(const std::vector<SampleRegionSetting>& regions, int activeRegion);
    void setPlayheadSeconds(double seconds);
    void setTool(Tool nextTool);
    [[nodiscard]] Tool currentTool() const { return tool; }
    [[nodiscard]] DrawAction drawAction() const { return drawActionMode; }
    void setDrawAction(DrawAction action);
    [[nodiscard]] PitchScope pitchScope() const { return pitchScopeMode; }
    void setPitchScope(PitchScope scope);
    [[nodiscard]] std::vector<ResolvedConnectionPitchCurve> diagnosticConnectionPitchCurves() const;
    [[nodiscard]] std::optional<juce::Point<float>> diagnosticConnectionPitchPoint(const juce::String&, int) const;
    bool diagnosticConnectionPitchShape(const juce::String&, int, PitchCurveShape);
    [[nodiscard]] bool diffSingerPitchRestoreAvailable() const;
    [[nodiscard]] bool diffSingerPitchRestoreMode() const { return dsPitchRestoreMode; }
    void setDiffSingerPitchRestoreMode(bool enabled);
    [[nodiscard]] bool diffSingerPitchOffsetAvailable() const;
    [[nodiscard]] bool diffSingerPitchOffsetMode() const { return dsPitchOffsetMode; }
    void setDiffSingerPitchOffsetMode(bool enabled);
    bool cancelDiffSingerPitchOffsetGesture();
    std::function<void()> onPitchPointModeChanged;
    // The flag lane: the same strip the loudness envelope uses, with g on a
    // fixed -50..+50 scale.  That is the range the engine clamps g to, so a
    // handle cannot be dragged somewhere that would not be rendered.
    // Which flag the lane is showing, and the button that changes it.
    [[nodiscard]] juce::String flagLaneFlag() const { return laneFlag; }
    enum class FlagEditMode { perNote, continuous };
    [[nodiscard]] FlagEditMode flagEditMode() const { return flagMode; }
    void setFlagEditMode(FlagEditMode mode);
    void setFlagLaneFlag(const juce::String& flag);
    void setDiffSingerFlagContext(bool enabled, const juce::var& capabilities);
    void setHifisamplerFlagContext(bool enabled);
    bool setDiffSingerParameterLayer(bool actual);
    [[nodiscard]] bool hasDiffSingerParameters() const;
    [[nodiscard]] juce::Rectangle<float> flagLaneLayerBounds(bool actual) const;
    [[nodiscard]] juce::Rectangle<float> flagLaneOffsetEnableBounds() const;
    std::function<void()> onDiffSingerParameterLaneRequested;
    bool flagLaneParameterSupported(const juce::String& flag) const;
    juce::StringArray unsupportedDiffSingerFlagCurves() const;
    [[nodiscard]] juce::Rectangle<float> flagLaneSwitchBounds() const;
    // Vertical zoom.  b spans -20..100 and g spans -50..50, so at the full
    // view a few units are a pixel or two and fine work is guesswork.  The
    // lane shows a slice of the range instead, and the slice follows a handle
    // dragged past its edge, so zooming in never puts a value out of reach.
    [[nodiscard]] std::pair<float, float> flagLaneWindow() const;
    [[nodiscard]] float flagLaneZoom() const { return laneZoom; }
    void nudgeFlagLaneZoom(bool zoomIn);
    [[nodiscard]] juce::Rectangle<float> flagLaneZoomButtonBounds(bool zoomIn) const;
    [[nodiscard]] juce::Rectangle<float> flagLaneBounds() const;
    // What the lane shows for a note: the curve it has stored for the flag on
    // show, or the flat one it would start from.  Switching the lane to
    // another flag is not an edit, so nothing is written until a handle is
    // actually moved -- but every flag looks ready to draw, rather than only
    // whichever one happened to be seeded.
    [[nodiscard]] std::vector<FlagCurvePoint> flagLaneCurveFor(
        const NoteData& note) const;
    // The stretch of a note, in the note's own time, that the flag on show can
    // reach: the whole note for most, and the onset alone -- from the lead-in
    // ahead of the note up to the note's start -- for b and bh.
    [[nodiscard]] std::pair<double, double> flagLaneSpanFor(
        const NoteData& note) const;
    // The stretch the curve is actually drawn across, in absolute seconds: the
    // note's sounding span, cut to the onset for b and bh.  A handle outside it
    // is drawn at the edge, so this is what says where a handle can be grabbed.
    [[nodiscard]] std::pair<double, double> flagLaneDrawnSpan(
        const NoteData& note, double absoluteStart) const;
    // Where a handle is drawn, and therefore where it answers a click.
    [[nodiscard]] juce::Point<float> diagnosticFlagHandleCentre(
        const juce::String& noteId, int index) const;
    // The notes a context menu acts on: the selection, or the note under the
    // cursor when nothing is selected.
    [[nodiscard]] std::vector<juce::String> chosenNoteIds(
        const juce::String& noteId) const;
    // Whether "reset linear flags" has anything to do: at least one of these
    // notes has the switch on and something drawn on it.  A selection mixing
    // switched-on and switched-off notes still offers it; the reset then
    // touches only the switched-on ones.
    [[nodiscard]] bool flagResetAvailable(
        const std::vector<juce::String>& noteIds) const;
    // Whether 恢复为音源OTO has anything to do: at least one of these notes has
    // an oto of its own.  The rest of a mixed selection is left as it is.
    [[nodiscard]] bool ownOtoRestoreAvailable(
        const std::vector<juce::String>& noteIds) const;
    // Whether one note's curve for the flag on show can be reset: only when it
    // has one.  A note showing the starting handle has nothing stored, so
    // there is nothing to undo.
    [[nodiscard]] bool flagResetAvailableForNote(const juce::String& noteId) const;
    // Every note the lane is showing -- the notes of the track in view.
    [[nodiscard]] std::vector<juce::String> laneNoteIds() const;
    // Whether any of them has a curve on the flag being shown.
    [[nodiscard]] bool flagResetAvailableForLane() const;
    void showFlagLaneSwitchContextMenu(juce::Point<int> screenPosition);
    [[nodiscard]] const TrackData* trackForNote(const juce::String& noteId) const;
    // Whether this note's 线性flag is in play: the switch is on and its track is
    // in a mode that offers curves at all.  A note keeps its curve when the
    // track moves to plain UTAU, and there the lane leaves it alone.
    [[nodiscard]] bool flagCurveActiveFor(const juce::String& noteId) const;
    [[nodiscard]] juce::Rectangle<float> flagLanePlotBounds() const;
    [[nodiscard]] float flagLaneY(float value) const;
    [[nodiscard]] float flagValueFromLaneY(float y) const;
    // The loudness lane's two zoom buttons.  The minus is U+2212, not the
    // ASCII hyphen, so it has to be decoded as UTF-8 rather than handed to
    // juce::String as bytes -- that is what drew a box beside the plus.
    // Pure, so what the button will draw can be checked without a window.
    [[nodiscard]] static juce::String amplitudeZoomLabel(bool zoomIn);
    // What a handle goes back to when it is reset: no shift at all, which is
    // what a curve is seeded with.
    static constexpr float flagCurveDefault = 0.0f;
    bool diffSingerFlagContext = false;
    bool hifisamplerFlagContext = false;
    juce::var diffSingerFlagCapabilities;
    // Which items the handle's menu offers.  The last handle cannot be
    // deleted -- a curve with none has nothing to say, and turning the switch
    // off is what says "no curve" -- and a handle already at the default has
    // nothing to reset to.  Pure, so the rule can be checked without a window.
    struct FlagPointMenuState { bool canDelete = false; bool canReset = false; };
    [[nodiscard]] static FlagPointMenuState flagPointMenuState(int pointCount,
                                                               float value);
    // Which keys carry their name, at a given row height.  Every key when
    // there is room; naturals only once the rows are too short for twelve
    // names an octave; C alone when even those would collide.  Pure, so the
    // rule can be checked without a window.
    [[nodiscard]] static bool keyboardLabelVisible(int midi, float rowHeight);
    // Where a spliced boundary crosses, measured against this note's start:
    // the instant the previous note stops sounding, which the rise has to
    // reach, and the instant the next note starts, where the fall has to
    // begin.  Absent when that side is not spliced, or when the crossing
    // would not fit inside the note.
    struct SpliceCrossing
    {
        std::optional<double> riseEnd;
        std::optional<double> fallStart;
    };
    [[nodiscard]] SpliceCrossing spliceCrossingFor(
        const NoteData& note, double absoluteStart,
        double firstTime, double lastTime) const;
    void selectAllNotes();
    [[nodiscard]] std::size_t diagnosticSelectedCount() const { return selectedNotes.size(); }
    void clearNoteSelection();
    void setSelectedNoteIds(const std::vector<juce::String>& noteIds);
    [[nodiscard]] std::vector<juce::String> selectedNoteIds() const;
    [[nodiscard]] bool canEditSelectedNativeVibrato() const;
    void showSelectedNativeVibratoDialog();
    // Removes the current selection, as the Delete key does.
    void deleteSelectedNotes();
    // When the selection begins and ends in the piece, lead-ins and tails
    // included.  Nothing selected, no answer.
    [[nodiscard]] std::optional<juce::Range<double>> selectedNotesTimeSpan() const;
    [[nodiscard]] int pixelForSeconds(double seconds) const;
    [[nodiscard]] double secondsForPixel(int pixel) const;
    void paint(juce::Graphics& g) override;
    // Re-apply Palette-derived colours held by child editors on a theme switch.
    void lookAndFeelChanged() override;
    void mouseMove(const juce::MouseEvent& event) override;
    void mouseExit(const juce::MouseEvent& event) override;
    void mouseDown(const juce::MouseEvent& event) override;
    void mouseDoubleClick(const juce::MouseEvent& event) override;
    void mouseDrag(const juce::MouseEvent& event) override;
    void mouseUp(const juce::MouseEvent& event) override;
    bool keyPressed(const juce::KeyPress& key) override;
    void focusLost(FocusChangeType) override;
    std::function<void(double)> onSeek;
    // Where an edit just began.  Distinct from onSeek, which is a transport
    // move the user asked for outright: this one follows the editing and is
    // expected to stand aside while something is playing.
    std::function<void(double)> onEditPosition;
    // Saved vibrato presets, as lines of "name,length,cycle,depth,fadein,
    // fadeout,phase,offset".  The roll has no preferences of its own.
    std::function<juce::String()> onLoadVibratoPresets;
    std::function<void(const juce::String&)> onSaveVibratoPresets;
    // The numbers of one preset line, name dropped.  Everything the dropdown
    // does with a preset goes through these two, so they can be checked
    // without a dialog on screen.
    // The presets that ship with the editor, in the same one-line form.
    [[nodiscard]] static juce::StringArray vibratoBuiltInPresets();
    [[nodiscard]] static juce::StringArray vibratoPresetValues(const juce::String& line);
    // The saved list with this preset added, replacing any of the same name --
    // saving twice under one name is asking to overwrite it, not to collect
    // two of them.
    [[nodiscard]] static juce::String vibratoPresetsWith(
        const juce::String& saved, const juce::String& name,
        const juce::StringArray& values);
    std::function<void(const juce::String&)> onNoteSelected;
    std::function<void(const juce::String&)> onEditRejected;
    std::function<void(const juce::String&)> onNoteAliasCommitted;
    std::function<void(const juce::String&, const juce::String&)> onRequestMaterialSourceInput;
    bool diagnosticBeginAliasEdit(const juce::String& noteId);
    bool suspendInlineAliasInput(const juce::String& noteId);
    void resumeInlineAliasInput(const juce::String& noteId, bool discard = false);
    // Open the four-region / oto editor for this note's voicebank entry.
    std::function<void(const juce::String&)> onOpenRegionEditor;
    std::function<void()> onOpenNativeEnvelope;
    std::function<void(const juce::String&)> onDiffSingerPronunciation, onDiffSingerPitch, onDiffSingerParameters;
    // 单独OTO编辑: the OTO editor, for this note's own copy of its entry.
    std::function<void(const juce::String&)> onOpenNoteOtoEditor;
    std::function<void(int, const SampleRegionSetting&, bool)> onSampleRegionEdited;

private:
    friend bool runPointGroupSmoke(const juce::File&);
    bool groupPointHitTest=false;
    // Selection outlives the drag. A consumed/stale selection still owns Delete
    // until another click, so key repeat cannot delete the containing note.
    struct PointSelection
    {
        enum Kind { pitch, amplitude, flag, cvvc, connectionPitch };
        juce::String noteId; // tagged owner key: connection ID only for connectionPitch
        int index = -1;
        std::uint64_t revision = 0;
        Kind kind = pitch;
        int part = 0;
        juce::String key;
    };
    std::optional<PointSelection> selectedPoint;
    std::vector<PointSelection> selectedPoints;
    struct PointCurveDraft
    {
        PointSelection identity;
        ProjectModel::PointCurveEdit edit;
        std::vector<int> indices;
        std::vector<std::pair<double,double>> original;
        double start=0, length=1, base=1, first=0, last=0, low=0, high=127, separation=.001;
        bool fixedEnds=false;
    };
    struct PointGroupGesture
    {
        std::vector<PointCurveDraft> curves;
        juce::Point<float> down;
        std::uint64_t revision=0;
        bool changed=false;
    };
    std::optional<PointGroupGesture> pointGroupGesture;
    void mouseDownSingle(const juce::MouseEvent&);
    std::vector<PointCurveDraft> pointCurveDrafts();
    void movePointGroup(const juce::MouseEvent&);
    void commitPointGroup(std::vector<PointCurveDraft>, std::uint64_t);
    void editSelectedPoints(bool initialize);
    void cancelPointDrag();
    void clearPointSelection();
    bool deletePitchAnchor(const juce::String& noteId, int index);
    bool deleteEnvelopePoint(const PointSelection& point);
    bool pointIsSelected(PointSelection::Kind kind, const juce::String& noteId,
                         int index, int part = 0, const juce::String& key = {}) const;
    bool envelopePointHasMoved = false;
    struct PointSelectionMouseListener : juce::MouseListener
    {
        explicit PointSelectionMouseListener(PianoRollComponent& roll) : owner(roll) {}
        void mouseDown(const juce::MouseEvent& event) override
        {
            if (event.eventComponent != &owner) owner.clearPointSelection();
        }
        PianoRollComponent& owner;
    } pointSelectionMouseListener { *this };
    struct NoteHit
    {
        juce::String id;
        // The real sounding stretch: it reaches back before the note for the
        // lead-in and can stop before the note's written end.
        juce::Rectangle<float> bounds;
        // The block that is drawn for the note -- the grid rectangle the
        // orange duration line marks out.  Pointing at a note is done with
        // this one, so a note is picked by what is on screen as the note
        // rather than by a lead-in lying over its neighbour.
        juce::Rectangle<float> nominalBounds;
        float midi = 60.0f;
        double startSeconds = 0.0;
        double durationSeconds = 0.0;
        double clipStartSeconds = 0.0;
    };

    struct PositionedUtauNote
    {
        juce::String id;
        double startSeconds = 0.0;
        double endSeconds = 0.0;
        bool automaticTransition = true;
    };

    struct ConsonantHandleInfo
    {
        juce::String noteId;
        double absoluteStart = 0.0;
        double scaledPreutterance = 0.0;
        double unscaledPreutterance = 0.0;
        double minimumPreutterance = 0.0;
        double maximumPreutterance = 0.0;
        double currentPreutterance = 0.0;
        int currentVelocity = 100;
        // A first region 谋 does not call a consonant: the consonant velocity
        // does not reach it, so the drag sets the lead-in directly and is free
        // between the note in front of it and its own start.  A consonant one
        // still speaks through the velocity, the way 界 always has.
        bool setsPin = false;
        double overlapSeconds = 0.0;
        std::function<double(int)> leadForVelocity;
    };

    // Vibrato is shaped by dragging its own trace rather than by typing
    // numbers: the handles sit on the swing itself, the way UTAU does it.
    enum class VibratoHandle { none, length, fadeIn, fadeOut, depth, cycle, offset, end };
    struct VibratoHandleInfo
    {
        juce::String noteId;
        VibratoHandle which = VibratoHandle::none;
        double absoluteStart = 0.0;
    };
    // Where a handle sits on the swing: seconds from the note's start, and
    // cents away from its nominal pitch.  Drawing and hit testing share this so
    // a handle is always grabbed exactly where it is drawn.
    // pixelOffsetX lifts a handle out of the note timeline.  The centre
    // offset handle uses it to sit in the empty column just before the note:
    // every position inside the note coincides with another handle at some
    // parameter setting, and two stacked dots can be neither told apart nor
    // grabbed reliably.
    struct VibratoHandlePlace { double timeSeconds; double cents; double pixelOffsetX; };
    [[nodiscard]] static VibratoHandlePlace vibratoHandlePlace(const NoteData& note,
                                                               VibratoHandle which);
    // Parameters to draw a note with: the live drag preview when it is the one
    // being dragged, otherwise what the note stores.
    [[nodiscard]] NoteData effectiveVibrato(const NoteData& note) const;
    [[nodiscard]] std::optional<VibratoHandleInfo> vibratoHandleAt(
        juce::Point<float> position) const;
    // Where a vibrato handle is drawn and grabbed, on screen.  The one place
    // both read it from, since the end handle steps off the line where the
    // fade-out handle would otherwise sit on top of it.
    [[nodiscard]] juce::Point<float> vibratoHandleCentre(const NoteData& shown,
                                                         VibratoHandle which,
                                                         double absoluteStart) const;

    // One of the three draggable four-region boundaries drawn inside a note
    // in the Jie/UTAU mode.
    struct JieSplitHandleInfo
    {
        juce::String noteId;
        int boundary = 0;                 // 0 = onset|glide .. 2 = nucleus|coda
        double spanStart = 0.0;           // absolute seconds
        double spanEnd = 0.0;
        std::array<double, 3> fractions {};
        int regions = 4;
    };

    enum class RegionHandle { none, start, fixedEnd, alignment, end };
    struct RegionHandleHit
    {
        int region = -1;
        RegionHandle handle = RegionHandle::none;
        float x = 0.0f;
    };

    void changeListenerCallback(juce::ChangeBroadcaster*) override;
    // What a release does with whatever drag is in progress.  mouseUp is
    // the only caller in the app; the offline checks reach it directly.
    void finishDrag();
    // Where a note-front drag has reached.  In 谋 this is the lead-in itself;
    // everywhere else it is expressed as a consonant velocity, which is what
    // the release then writes.
    void dragConsonantTo(double cursorSeconds);
    void rebuildLayout();
    NoteDanceAnimation noteDance;
    void rebuildUtauSoundSpans();
    void requestCvvcLayouts();
    juce::ThreadPool cvvcLayoutPool {1};
    std::set<std::uint64_t> cvvcLayoutRequests;
    // Boundaries that have to end up where they already are, once something
    // has moved the span they are measured against.  Filled by resetConsonant
    // and spent on the next layout, when the new spans exist.
    struct StandingSplit { juce::String id; double first = 0.0, second = 0.0, third = 0.0; };
    std::vector<StandingSplit> pendingSplitRestate;
    void restateStandingSplits();
    void rebuildNoteHits();
    [[nodiscard]] std::optional<std::pair<juce::String,juce::String>> nativeJoinAt(juce::Point<float>) const;
    [[nodiscard]] bool clipIsVisible(const TrackData& track, const ClipData& clip) const;
    void pruneHiddenNoteSelection();
    [[nodiscard]] const NoteHit* resizableTailAt(juce::Point<float> position,
                                                 double& maximumDuration) const;
    // The split currently in effect for a note: the manual one if set,
    // otherwise the allocation the renderer would perform.  Empty when the
    // sample has no four-region data.
    // leadInOverride is for a lead-in being dragged: the block on screen is
    // already drawn against it, so the boundaries inside must be too.
    // regionsOut, when asked for, is how many regions the note's oto entry
    // has: 谋 says so per entry, everything else has four.  The fractions past
    // that are 1.0 -- there is no boundary there to draw or to grab.
    [[nodiscard]] std::optional<std::array<double, 3>> jieFractionsFor(
        const TrackData& track, const NoteData& note, double spanSeconds,
        std::optional<double> leadInOverride = {},
        int* regionsOut = nullptr) const;
    // midiToY() returns the top of a row, but every pitch anchor and curve is
    // drawn at midiToY() + rowHeight/2, the row centre.  Converting a cursor
    // position back with plain yToMidi() therefore places the point half a row
    // below the pointer; this is the matching inverse.
    [[nodiscard]] float pitchMidiFromY(float y) const
    {
        return yToMidi(y - rowHeight * 0.5f);
    }
    [[nodiscard]] std::optional<JieSplitHandleInfo> jieSplitHandleAt(
        juce::Point<float> position) const;
    [[nodiscard]] std::optional<ConsonantHandleInfo> consonantHandleAt(
        juce::Point<float> position) const;
    [[nodiscard]] double noteEditQuantumSeconds() const;
    [[nodiscard]] double nativeEdgeDragDelta(double originalSeconds, double travelSeconds, bool free) const;
    [[nodiscard]] int noteEditDivision() const;
    void beginInlineAliasEdit(const NoteHit& hit);
    void finishInlineAliasEdit(bool accept);
    // Tab in the lyric box: keep this lyric and carry on at the next note on
    // the track (Shift+Tab, the one before).
    void advanceInlineAliasEdit(bool forward);
    void commitInlineAlias(const juce::String& noteId, juce::String alias);
    [[nodiscard]] juce::String adjacentNoteOnTrack(const juce::String& noteId,
                                                   bool forward) const;
    void revealInViewport(juce::Rectangle<float> area);
    void updateCanvasSize();
    void drawClipWaveforms(juce::Graphics& g);
    [[nodiscard]] float timeToX(double seconds) const;
    [[nodiscard]] float midiToY(float midi) const;
    [[nodiscard]] float yToMidi(float y) const;
    [[nodiscard]] double gridQuarterNotes() const;
    [[nodiscard]] double gridSecondsAt(double seconds) const;
    [[nodiscard]] double snapToGrid(double seconds) const;
    // Snapping for a note being made, as opposed to one being moved: the note
    // belongs to the cell that was clicked, so this floors instead of
    // rounding.  Rounding sent a click in the right half of a cell into the
    // next one, which put two clicks either side of a line in the same place.
    [[nodiscard]] double snapDownToGrid(double seconds) const;
    // A sixty-fourth of a bar, which is the step a drawn note grows by.
    [[nodiscard]] double drawUnitSecondsAt(double seconds) const;
    // True when the track being drawn on is sung by UTAU, which is where
    // drawing makes notes rather than pitch curves.  Asked of the track, not
    // of a clip: a track that has none yet is still a UTAU track, and gating
    // on the focused clip sent the first stroke on a new track down the wrong
    // branch.
    [[nodiscard]] bool canDrawNotesOnFocusedTrack() const;
    void beginDrawingNote(const juce::MouseEvent& event);
    // The clip drawing goes into, making one on the focused track when it has
    // none.  Returns an empty id when there is nowhere to draw.
    juce::String clipForDrawingAt(double seconds);
    [[nodiscard]] const NoteData* findNote(const juce::String& noteId,
                                           bool* isUtau = nullptr) const;
    [[nodiscard]] std::vector<PositionedUtauNote> positionedUtauNotesFor(
        const juce::String& noteId) const;
    [[nodiscard]] std::optional<PositionedUtauNote> previousUtauNoteFor(
        const juce::String& noteId) const;
    [[nodiscard]] std::optional<PositionedUtauNote> nextUtauNoteFor(
        const juce::String& noteId) const;
    [[nodiscard]] static bool formsAdjacentPitchBoundary(
        const PositionedUtauNote& left, const PositionedUtauNote& right);
    // A native pitch join: a connected note whose displayed line must start at
    // the previous note's tail pitch and glide to its own over a short window,
    // exactly as the renderer blends the two so what is drawn is what is heard.
    // Empty for a note with no incoming connection or no adjacent predecessor.
    struct IncomingJoinGlide { double leadMidi = 60.0; double joinSeconds = 0.0; };
    [[nodiscard]] std::optional<IncomingJoinGlide> incomingJoinGlideFor(
        const NoteData& note);
    std::vector<PitchCurveEditPoint>& pitchAnchorsFor(const NoteData& note);
    [[nodiscard]] juce::PopupMenu buildPitchCurveShapeMenu(const juce::String& noteId,
                                                           int anchorIndex);
    void showPitchCurveShapeMenu(const juce::String& noteId, int anchorIndex,
                                 juce::Point<int> screenPosition);
    // The menu itself, so a check can walk what was actually built rather
    // than only the list it was supposed to be built from.
    struct NoteMenu
    {
        juce::PopupMenu menu;
        double splitSeconds = 0.0;
        std::vector<juce::String> mergeIds;
    };
    [[nodiscard]] NoteMenu buildNoteMenu(const juce::String& noteId) const;
    void showNoteContextMenu(const juce::String& noteId,
                             juce::Point<int> screenPosition);
    void showNoteTimingDialog(const juce::String& noteId);
    void showVibratoDialog(const juce::String& noteId);
    void confirmBakeVibrato(const juce::String& noteId);
    void showRegionFlagDialog(const juce::String& noteId);
    // A silence opened in front of a note.  The quarter bar is the one-click
    // amount; the dialog counts in units of a 128th of a quarter note, which
    // is the smallest step either offers.
    void showGapDialog(const juce::String& noteId);
    // Asks for an STP in milliseconds and gives it to every selected note.
    void showStpDialog(const juce::String& noteId);
    void showEnvelopeBaseDialog(const juce::String& noteId);
    // A UTAU note's overlap as it stands: its own pin, else its entry's.
    [[nodiscard]] double effectiveUtauOverlapFor(const juce::String& noteId) const;
    // A silence with a note on either side of it.  Right-clicking one offers
    // to close it up or to open a note into it; both move what follows.
    struct GapInfo
    {
        juce::String clipId;
        juce::String previousId, nextId;
        double fromSeconds = 0.0, toSeconds = 0.0;   // absolute
        float midi = 60.0f;                          // the note in front of it
    };
    [[nodiscard]] std::optional<GapInfo> gapAt(juce::Point<float> position) const;
    void showGapContextMenu(const GapInfo& gap, juce::Point<int> screenPosition);
    [[nodiscard]] double gapUnitSecondsAt(double seconds) const;
    [[nodiscard]] double quarterBarSecondsAt(double seconds) const;
    // Absolute start of a note, or -1 when it is not in the piece.
    [[nodiscard]] double absoluteStartOf(const juce::String& noteId) const;
    // Whether this note sits on a UTAU track.  The gap items are offered
    // in every UTAU mode and nowhere else.
    [[nodiscard]] bool isUtauNote(const juce::String& noteId) const;
    // Opens a quarter bar in front of the gap's following note and puts a note
    // of that length in the space, carrying on from the one before it.
    juce::String insertNoteIntoGap(const GapInfo& gap);
    // One box per region the note's oto entry really has, named the way the
    // roll names them: the parts of a syllable in 界, by number in 谋, where
    // any region may be a consonant and there may be two or three of them.
    struct RegionFlagField
    {
        juce::String key;
        juce::String label;
        juce::String value;
    };
    [[nodiscard]] std::vector<RegionFlagField> regionFlagFieldsFor(
        const juce::String& noteId) const;
    void showTransposeDialog(const juce::String& noteId);
    void drawBarNumbers(juce::Graphics& g);
    void showCustomBezierDialog(const juce::String& noteId, int anchorIndex);
    // Asks for one anchor's frequency in hertz and moves it there.
    void showAnchorFrequencyDialog(const juce::String& noteId, int anchorIndex);
    [[nodiscard]] static float pitchAt(const std::vector<PitchCurveEditPoint>& anchors,
                                       double timeSeconds);
    [[nodiscard]] std::vector<AmplitudeEnvelopePoint> amplitudeEnvelopeFor(
        const NoteData& note, double absoluteStart) const;
    [[nodiscard]] std::vector<AmplitudeEnvelopePoint> effectiveAmplitudeEnvelope(const NoteData& note,double absoluteStart,const std::vector<AmplitudeEnvelopePoint>& base) const;
    [[nodiscard]] static float amplitudeDbAt(
        const std::vector<AmplitudeEnvelopePoint>& points, double timeSeconds);
    // One millisecond of a note's picture: the piece as the mix fades it,
    // shaped by the envelope on screen.  The note's row and the envelope lane
    // both draw from this.
    [[nodiscard]] static float notePictureAt(const UtauNoteWaveform& waveform,
                                             const std::vector<AmplitudeEnvelopePoint>& envelope,
                                             std::size_t bucket, bool high);
    // And the faint layer behind the lane, as a level: the same piece with
    // the envelope's shape left out between its first and last points.
    [[nodiscard]] static float noteGhostAt(const UtauNoteWaveform& waveform,
                                           const std::vector<AmplitudeEnvelopePoint>& envelope,
                                           std::size_t bucket);
    [[nodiscard]] static bool withinEnvelope(const UtauNoteWaveform& waveform,
                                             const std::vector<AmplitudeEnvelopePoint>& envelope,
                                             std::size_t bucket);
    [[nodiscard]] float amplitudeY(float midi, float gainDb) const;
    [[nodiscard]] float amplitudeDbFromY(float midi, float y) const;
    [[nodiscard]] juce::Rectangle<float> amplitudeLaneBounds() const;
    [[nodiscard]] juce::Rectangle<float> amplitudeLanePlotBounds() const;
    [[nodiscard]] float amplitudeLaneY(float gainDb) const;
    [[nodiscard]] float amplitudeDbFromLaneY(float y) const;
    [[nodiscard]] juce::Rectangle<float> amplitudeZoomButtonBounds(bool zoomIn) const;
    [[nodiscard]] static float amplitudePercentFromDb(float gainDb);
    [[nodiscard]] static float amplitudeDbFromPercent(float percent);
    [[nodiscard]] std::vector<AmplitudeEnvelopePoint> mapAmplitudeEnvelopeToNote(
        const std::vector<AmplitudeEnvelopePoint>& source,
        const juce::String& sourceNoteId, const juce::String& targetNoteId) const;
    void commitAmplitudeEnvelopeToSelection(
        const juce::String& sourceNoteId,
        const std::vector<AmplitudeEnvelopePoint>& source);

    struct CvvcBoundaryHandle
    {
        juce::String noteId;
        int part=0;
        juce::Rectangle<float> bounds;
    };
    struct CvvcBoundaryDrag
    {
        juce::String noteId;
        int part=0;
        double duration=0, original=0, position=0, from=0, to=0;
        std::uint64_t revision=0;
        std::vector<backend::UtauPhonemeSpan> spans;
    };
    std::optional<CvvcBoundaryDrag> cvvcBoundaryDrag, pendingCvvcBoundary;
    [[nodiscard]] std::vector<CvvcBoundaryHandle> cvvcBoundaryHandles() const;
    [[nodiscard]] std::optional<CvvcBoundaryHandle> cvvcBoundaryAt(juce::Point<float>) const;
    bool cvvcBoundaryMouseDown(const juce::MouseEvent&);
    void cvvcBoundaryMouseDrag(const juce::MouseEvent&);
    void finishCvvcBoundaryDrag();
    void applyCvvcBoundaryPreview();
    struct CvvcLanePiece
    {
        juce::String noteId, token, key;
        int part = 0;
        double from = 0, to = 0;
        backend::CvvcEnvelopeCurve points;
        backend::UtauMixEnvelope mixEnvelope; // absolute timeline, read-only renderer data
    };
    struct CvvcLaneDrag
    {
        CvvcLanePiece piece;
        int point = -1;
        std::uint64_t revision = 0;
        bool changed = false;
        bool continuous = false;
    };
    std::optional<CvvcLaneDrag> cvvcLaneDrag;
    juce::String cvvcSelectedNote;
    int cvvcSelectedPart = 0;
    struct CvvcPhonemeTab { juce::String noteId, token; int part; juce::Rectangle<float> bounds; };
    [[nodiscard]] std::vector<CvvcPhonemeTab> cvvcPhonemeTabs() const;
    void drawCvvcPhonemeTabs(juce::Graphics&) const;
    bool cvvcPhonemeTabMouseDown(const juce::MouseEvent&);
    [[nodiscard]] bool cvvcTimingEnabled(const juce::String& noteId) const;
    [[nodiscard]] bool usesCvvcEnvelope(const juce::String& noteId) const;
    [[nodiscard]] std::vector<CvvcLanePiece> cvvcLanePieces(bool amplitude) const;
    int drawCvvcEnvelopeLane(juce::Graphics&, bool amplitude) const;
    bool cvvcEnvelopeMouseDown(const juce::MouseEvent&, bool addPoint);
    void cvvcEnvelopeMouseDrag(const juce::MouseEvent&);
    void finishCvvcEnvelopeDrag();

    ProjectModel& model;
    const I18n& strings;
    ProjectData snapshot;
    juce::AudioFormatManager formats;
    juce::AudioThumbnailCache thumbnailCache { 96 };
    std::unordered_map<std::string, std::unique_ptr<juce::AudioThumbnail>> thumbnails;
    std::unique_ptr<NativePitchVoicingCache> consonantVoicing;
    void requestConsonantVoicing();
    // Dragging used to repaint the whole window at mouse-move rate.  Only a
    // vertical band actually changes, and these work out which one.
    [[nodiscard]] juce::Range<float> dragRepaintBand(float mouseX) const;
    void repaintDrag(float mouseX);
    // What the last frame of this drag repainted.  A marquee dragged back on
    // itself shrinks, and without covering where it was the outline it drew
    // last frame would be left behind.
    juce::Range<float> lastDragBand;
    // Where the playhead stood when this press began.  A press moves it, but a
    // press that turns out to be a marquee gives it back.
    double playheadBeforeGesture = 0.0;
    // Whether this press has travelled far enough to count as a drag.  A
    // double click is two presses with the pointer never quite still, and
    // note geometry used to follow that wobble.
    bool noteDragPassedThreshold = false;
    juce::Rectangle<int> playheadBand;
    bool cullOffscreenNotes = true;
    std::unordered_map<std::string, std::pair<double, double>> utauSoundSpans;
    bool readsVoicebankInBackground = false;
    // A layout is already due when a bank finishes reading.
    mutable bool awaitingVoicebank = false;
    // Whether a track's bank can be asked about without waiting for it to be
    // read.  Always, unless reading in the background; when it cannot, the
    // reading is under way and a layout follows it.
    [[nodiscard]] bool voicebankReadFor(const TrackData& track) const;
    // Who a note sits between, worked out once with the spans.  Asking for it
    // per note meant rebuilding and sorting the whole track's note list twice
    // for every note drawn, which turned one paint into quadratic work.
    struct UtauNeighbours
    {
        std::string previous;
        std::string next;
        bool nextSpliced = false;
    };
    std::unordered_map<std::string, UtauNeighbours> utauNeighbours;
    std::vector<NoteHit> noteHits;
    class InlineAliasEditor final : public juce::TextEditor
    {
    public:
        std::function<void()> requestSource;
        bool keyPressed(const juce::KeyPress& key) override
        {
            if(key.isKeyCode(juce::KeyPress::returnKey)&&key.getModifiers().isCtrlDown()&&requestSource)
            {requestSource();return true;}
            return juce::TextEditor::keyPressed(key);
        }
    };
    InlineAliasEditor inlineAliasEditor;
    juce::String inlineAliasNoteId, suspendedAliasNoteId, suspendedAliasText;
    bool inlineAliasSuspended = false;
    std::uint64_t inlineAliasGeneration = 0, suspendedAliasGeneration = 0, suspendedAliasRevision = 0;
    std::vector<SampleRegionSetting> sampleRegions;
    std::vector<RegionHandleHit> regionHandleHits;
    int activeSampleRegion = -1;
    int draggedSampleRegion = -1;
    RegionHandle draggedRegionHandle = RegionHandle::none;
    float pixelsPerSecond = 140.0f;
    float rowHeight = 22.0f;
    int highestMidi = 96;
    int lowestMidi = 24;
    bool sourceEditMode = false;
    bool showPhonemeInfo = false;
    bool showNoteLabels = false;
    // The shared UTAU-style editor starts with waveform ink hidden; it can be
    // enabled from the view menu when source detail is useful.
    bool showWaveforms = false;
    bool showUtauWaveforms = false;
    bool showNativeWaveforms = true;
    bool showLyrics = true;
    bool showNativeRenderedWaveforms = false;
    std::shared_ptr<const std::vector<NativeRenderedWaveform>> nativeWaveforms;
    std::unordered_map<std::uint64_t, const NativeRenderedPeaks*> nativePeaksByHash;
    std::unordered_map<const ClipData*, std::uint64_t> nativeClipHashes;
    [[nodiscard]] bool nativeNoteTimingEnabled(const juce::String& id) const;
    void rebuildNativeWaveformHashes();
    [[nodiscard]] const NativeRenderedPeaks* nativeRenderedPeaksFor(const ClipData&) const;
    std::shared_ptr<const std::vector<UtauNoteWaveform>> utauWaveforms;
    std::unordered_map<std::string, std::vector<backend::UtauPhonemeSpan>> diffSingerPhonemes;
    std::unordered_set<std::string> diffSingerNoteIds;
    void drawUtauNoteWaveforms(juce::Graphics& g);
    void drawNativeNoteWaveforms(juce::Graphics& g);
    [[nodiscard]] juce::Rectangle<float> nativeWaveformLabelBounds(
        const TrackData&, const ClipData&, const NoteData&, juce::Rectangle<float>) const;
    void drawNativeAudioOverlaps(juce::Graphics& g);
    std::vector<std::pair<juce::Rectangle<float>,juce::String>> nativeOverlapHints;
    struct NativeOverlapSourceHit { juce::Rectangle<float> bounds;juce::String noteId; };
    std::vector<NativeOverlapSourceHit> nativeOverlapSourceHits;
    struct NativeOverlapSelectorState
    {
        juce::String targetNoteId, leftSourceId, rightSourceId;
        double targetSeconds = 0;
        std::uint64_t revision = 0;
    };
    struct NativeOverlapPair
    {
        const TrackData* track = nullptr;
        const ClipData* left = nullptr;
        const ClipData* right = nullptr;
        double from = 0, to = 0;
    };
    std::optional<NativeOverlapSelectorState> nativeOverlapSelector;
    bool showOverlap = false;
    std::vector<std::pair<juce::String,juce::Rectangle<float>>> nativeOverlapLines;
    std::size_t nativeOverlapCardCount = 0;
    juce::Rectangle<float> nativeOverlapSelectorClose;
    [[nodiscard]] std::optional<NativeOverlapPair> overlapPairFor(const juce::String& noteId, double time) const;
    void openNativeOverlapSelector(const juce::String& noteId, double time);
    void closeNativeOverlapSelector();
    void drawNoteHints(juce::Graphics& g) const;
    // The rendered audio behind the loudness envelope, in the lane's own
    // percent scale.  Returns how many notes it drew, which is what a check
    // can ask for without counting pixels.
    int drawAmplitudeLaneWaveforms(juce::Graphics& g);
    // The envelope a note is drawn with right now, which during a drag is the
    // one being dragged rather than the one the project still holds.
    [[nodiscard]] std::vector<AmplitudeEnvelopePoint> displayAmplitudeEnvelope(
        const NoteData& note, double absoluteStart) const;
    void showBatchLyricDialog(const juce::String& noteId);
    // The notes a batch of lyrics fills, in the order they are sung: the
    // selection when several are selected, otherwise this note and the ones
    // after it in its own clip.
    [[nodiscard]] std::vector<juce::String> batchLyricTargets(
        const juce::String& noteId) const;
    bool showNoteRange = true;
    bool showEnvelope = false;
    bool showPitchLine = true;
    bool showOriginalPitchLine = true;
    bool consonantPitchDashed = false;
    Tool tool = Tool::note;
    DrawAction drawActionMode = DrawAction::pitchCurve;
    PitchScope pitchScopeMode = PitchScope::focusedClip;
    [[nodiscard]] const TrackData* currentPitchTrack() const;
    [[nodiscard]] static std::pair<double,double> trackPitchEditDomain(const NoteData&);
    bool beginTrackPitchStroke(const juce::MouseEvent&);
    bool commitTrackPitchStroke();
    bool planTrackPitchStroke(std::vector<ProjectModel::PointCurveEdit>&, juce::String&);
    bool trackPitchStrokeActive = false;
    std::uint64_t trackPitchStrokeRevision = 0;
    juce::String trackPitchStrokeTrack;
    juce::String focusedClip;
    juce::String focusedTrack;
    double playheadSeconds = 0.0;
    juce::String selectedNote;
    std::unordered_set<std::string> selectedNotes;
    juce::String draggedNote;
    float dragStartMidi = 0.0f;
    float previewMidi = 0.0f;
    float dragStartY = 0.0f;
    bool finePitchDrag = false;
    bool nativeTrimDrag = false;
    enum class DragMode { none, pitch, moveUtauNote, moveNativeNote, resizeNativeLeft, resizeNativeRight,
                          resizeLeft, resizeRight, consonantLeadIn,
                          drawPitch, linePitch, drawNewNote,
                          pointPitch, amplitudePoint, flagPoint, continuousFlag, jieSplit, vibrato,
                          marquee } dragMode = DragMode::none;
    // A note being drawn: a rectangle under the pointer until the button
    // comes up, and only then a note.  Nothing is written to the project
    // while the drag runs, so letting go outside the clip simply leaves it.
    // Where the pointer last was over the roll, unsnapped.
    std::optional<double> hoverSeconds;
    juce::String drawClipId;
    double drawStartSeconds = 0.0;      // absolute
    double drawLengthSeconds = 0.0;
    double drawAvailableSeconds = 0.0;  // room before the next note or the clip end
    double drawUnitSeconds = 0.0;
    int drawLengthDivisionValue = 64;
    float drawMidi = 60.0f;
    VibratoHandle draggedVibratoHandle = VibratoHandle::none;
    double vibratoDragAbsoluteStart = 0.0;
    NoteData previewVibrato;
    int draggedJieBoundary = 0;
    double jieSpanStartSeconds = 0.0;
    double jieSpanEndSeconds = 0.0;
    std::array<double, 3> previewJieFractions {};
    // The count the dragged note has, so the paint draws the same number of
    // bands mid-drag as it did before the drag started.
    int previewJieRegions = 4;
    double dragStartSeconds = 0.0;
    double dragDurationSeconds = 0.0;
    double dragClipStartSeconds = 0.0;
    double previewStartSeconds = 0.0;
    double previewDurationSeconds = 0.0;
    double previewMoveDeltaSeconds = 0.0;
    double minimumMoveDeltaSeconds = 0.0;
    float minimumDragPitchDelta=-127.f, maximumDragPitchDelta=127.f;
    double resizeMaximumDurationSeconds = 1.0e12;
    double consonantNoteAbsoluteStart = 0.0;
    double consonantScaledPreutterance = 0.0;
    std::function<double(int)> consonantCvvcLead;
    int previewConsonantVelocityBeforeDrag = 100;
    juce::String pendingCvvcHead;
    std::uint64_t pendingCvvcHeadRevision = 0;
    double pendingCvvcHeadStart = 0;
    double consonantUnscaledPreutterance = 0.0;
    double consonantMinimumPreutterance = 0.0;
    double consonantMaximumPreutterance = 0.0;
    double previewConsonantPreutterance = 0.0;
    int previewConsonantVelocity = 100;
    bool consonantSetsPin = false;
    double consonantOverlapSeconds = 0.0;
    // Whether the pointer has gone anywhere since the lead-in handle was
    // pressed.  A press that never did is a click, and a release commits
    // nothing: committing released the note's pin, which put a note behind a
    // 拼字 note back to its entry's own timing every time its seam was clicked.
    bool consonantDragTravelled = false;
    // Hand shake on a click, not a drag.  Well under the six the note body
    // waits for, since a handle is grabbed on purpose and a small deliberate
    // nudge has to count.
    static constexpr int consonantClickSlopPixels = 3;
    double pitchEditAbsoluteStart = 0.0;
    std::vector<PitchCurveEditPoint> pitchStroke;
    bool pointPitchHasMoved = false;
    juce::Point<float> pointPitchPressPosition;
    bool pointDragCanMoveHorizontally = false;
    double pointDragMinimumTime = 0.0;
    double pointDragMaximumTime = 0.0;
    // On a shared pitch line: how far right any point of the dragged note may
    // go and still decide something.  Relative to the note's start.
    double pointDragOwnedUntil = std::numeric_limits<double>::infinity();
    std::unordered_map<std::string, std::vector<PitchCurveEditPoint>> pitchAnchorCache;
    // The pitch lines notes share, per track id; see sharedPitchLines.  While a
    // pitch point is being dragged, the dragged note's track is read with the
    // stroke standing in for that note's points, so the line follows the drag.
    // followDrag false reads the lines as they were before the drag began:
    // which handles are shown is decided by those, so none appear or vanish
    // while a point is being dragged.
    [[nodiscard]] const SharedPitchLines& sharedLinesFor(const TrackData& track,
                                                         bool followDrag = true) const;
    [[nodiscard]] const TrackData* trackOf(const juce::String& noteId) const;
    // The line as drawn for a note: the corners it passes through, relative to
    // the note's start, and its pitch at a time relative to the note's start.
    [[nodiscard]] std::optional<TransitionBridge> transitionBridge(
        const TrackData& track, const NoteData& note, double absoluteStart);
    [[nodiscard]] std::vector<double> pitchLineBreaks(const TrackData& track,
                                                      const NoteData& note,
                                                      double absoluteStart);
    [[nodiscard]] float pitchLineMidiAt(const TrackData& track, const NoteData& note,
                                        double absoluteStart, double time,
                                        const std::vector<PitchCurveEditPoint>* nativeDensePoints = nullptr);
    // Whether a handle of this note at this absolute moment decides anything:
    // on a shared line, only inside the stretch the note owns.
    [[nodiscard]] bool pitchHandleOffered(const NoteData& note, double absoluteSeconds) const;
    // The last point a note shows, and the first, in absolute seconds: where
    // the notes either side of it may bring their own points up to, and no
    // further.
    [[nodiscard]] std::optional<double> lastShownPitchPoint(const juce::String& noteId,
                                                            double absoluteStart);
    [[nodiscard]] std::optional<double> firstShownPitchPoint(const juce::String& noteId,
                                                             double absoluteStart);
    mutable std::map<juce::String, SharedPitchLines> sharedLineCache;
    mutable std::map<juce::String, SharedPitchLines> pitchPreviewSharedCache;
    mutable std::map<juce::String, std::vector<AudioEngine::ProjectedNotePitch>> protocolPitchCache, protocolPreviewCache;
    const AudioEngine::ProjectedNotePitch* protocolPitchFor(const TrackData&, const juce::String&) const;
    mutable std::map<juce::String, std::vector<ResolvedConnectionPitchCurve>> connectionPitchCache, connectionPreviewCache;
    std::optional<ProjectModel::PointCurvePlanPreview> pitchPlanPreview;
    std::vector<ProjectModel::PointCurveEdit> pitchPendingEdits;
    std::unordered_map<std::string, std::vector<PitchCurveEditPoint>> pitchPreviewAnchorCache;
    const TrackData& pitchPreviewTrack(const TrackData&) const;
    const NoteData& pitchPreviewNote(const NoteData&) const;
    const std::vector<ResolvedConnectionPitchCurve>& connectionCurvesFor(const TrackData&, bool preview = true) const;
    std::vector<ResolvedConnectionPitchCurve> visibleConnectionCurves(bool preview = true) const;
    bool updatePitchPlanPreview(const std::vector<ProjectModel::PointCurveEdit>&, std::uint64_t);
    void clearPitchPlanPreview();
    std::optional<std::pair<juce::String,int>> connectionPointAt(juce::Point<float>) const;
    std::optional<juce::String> connectionCurveAt(juce::Point<float>) const;
    void drawConnectionPitchCurves(juce::Graphics&);
    void clipNotePitchAgainstConnections(juce::Graphics&, const TrackData&) const;
    bool connectionPitchMouseDown(const juce::MouseEvent&);
    void moveConnectionPitchPoint(const juce::MouseEvent&);
    bool connectionPitchDoubleClick(const juce::MouseEvent&);
    juce::PopupMenu connectionPitchPointMenu(const juce::String&, int) const;
    bool applyConnectionPitchPointMenu(const juce::String&, int, int);
    void showConnectionPitchPointMenu(const juce::String&, int, juce::Point<int>);
    bool graphPointDragging = false, graphDragValid = false;
    juce::String graphDragId;
    int graphDragIndex = -1;
    juce::Point<float> graphDragDown;
    std::uint64_t graphDragRevision = 0;
    std::vector<PitchCurveEditPoint> graphDragOriginal;
    mutable std::map<juce::String, NativeNoiseRanges> nativePitchVisibilityCache;
    mutable std::map<juce::String, NativeNoiseRanges> nativePitchAuthoredCache;
    [[nodiscard]] const NativeNoiseRanges& nativePitchVisibilityFor(
        const ClipData&, const NoteData&) const;
    [[nodiscard]] const NativeNoiseRanges& nativePitchAuthoredFor(
        const ClipData&, const NoteData&) const;
    mutable std::map<juce::String, NativeSharedEnvelopes> nativeEnvelopeCache;
    [[nodiscard]] const NativeSharedEnvelopeMember* nativeEnvelopeMember(const juce::String& id) const;
    mutable std::uint64_t draggedSharedKey = 0;
    mutable SharedPitchLines draggedSharedLines;
    int draggedPitchAnchor = -1;
    bool dsPitchRestoreMode = false;
    bool dsPitchOffsetMode = false;
    std::set<juce::String> dsPitchOffsetNoteIds;
    juce::String offsetNoteId;
    std::vector<PitchCurveEditPoint> offsetStroke, offsetOriginal, offsetInk;
    juce::Point<float> offsetPress;
    double offsetAbsoluteStart = 0.0;
    double offsetLastTime = 0.0;
    float offsetLastValue = 0.0f;
    int offsetAnchor = -1;
    bool offsetWhole = false, offsetFreehand = false, offsetMoved = false;
    std::uint64_t offsetRevision = 0;
    std::vector<PitchCurveEditPoint> offsetCurveFor(const NoteData& note) const;
    float displayedPitchOffset(const NoteData& note, double time) const;
    void drawDiffSingerPitchOffsets(juce::Graphics& g) const;
    void offsetMouseDown(const juce::MouseEvent& event);
    void offsetMouseDrag(const juce::MouseEvent& event);
    void offsetMouseUp();
    void offsetDoubleClick(const juce::MouseEvent& event);
    bool addOffsetPoint(const juce::String& noteId, double localSeconds);
    void showOffsetMenu(const juce::String& noteId, int point, double localSeconds, juce::Point<int> screenPosition);
    FlagEditMode flagMode = FlagEditMode::perNote;
    struct ContinuousFlagTarget {
        double start, from, to;
        std::vector<FlagCurvePoint> original;
    };
    std::map<juce::String, ContinuousFlagTarget> continuousFlagTargets;
    std::map<juce::String, std::vector<FlagCurvePoint>> continuousFlagPreview;
    std::vector<FlagCurvePoint> continuousFlagInk;
    FlagCurvePoint continuousFlagLast;
    std::uint64_t continuousFlagRevision = 0;
    void beginContinuousFlag(juce::Point<float> position);
    void drawContinuousFlag(juce::Point<float> position);
    void cancelContinuousFlag();
    std::vector<FlagCurvePoint> flagStroke;
    int draggedFlagPoint = -1;
    double flagEditAbsoluteStart = 0.0;
    double flagPointMinimumTime = 0.0;
    double flagPointMaximumTime = 0.0;
    [[nodiscard]] juce::Path diffSingerPitchReferencePath(const NoteData& note,
                                                         double absoluteStart) const;
    void drawFlagLane(juce::Graphics& g) const;
    void showFlagLaneSwitchMenu();
    // Defaults to the formant shift, which is what a curve is usually wanted
    // for; every other flag the engine can vary per frame is a menu away.
    juce::String laneFlag { "g" };
    juce::String dsLastActualFlag { "DS:ABS:BREC" };
    bool dsPreferActual = true;
    [[nodiscard]] int flagLaneExtraControlWidth() const;
    [[nodiscard]] std::vector<juce::String> diffSingerOffsetTargets() const;
    [[nodiscard]] bool diffSingerOffsetsEnabled() const;
    // 1 is the whole range; above that the window is centred on laneCentre.
    float laneZoom { 1.0f };
    float laneCentre { 0.0f };
    void centreFlagLaneOnCurves();
    void keepFlagValueInView(float value);
    void showFlagPointMenu(const juce::String& noteId, int index,
                           juce::Point<int> screenPosition);
    void showFlagPointValueDialog(const juce::String& noteId, int index);
    // What that dialog opens showing, and empty when it would refuse to open.
    // The number a handle is carrying is the point of the dialog, so it is
    // worked out here and the dialog and the offline check read the same one.
    [[nodiscard]] juce::String flagPointValueText(const juce::String& noteId,
                                                  int index) const;
    // What that dialog says above the box: which flag is being edited, its
    // range, and which of the note's handles this is.
    [[nodiscard]] juce::String flagPointValueBlurb(const juce::String& noteId,
                                                   int index) const;
    void showFlagPointBezierDialog(const juce::String& noteId, int index);
    std::vector<AmplitudeEnvelopePoint> amplitudeStroke;
    int draggedAmplitudePoint = -1;
    double amplitudeEditAbsoluteStart = 0.0;
    float amplitudeEditMidi = 60.0f;
    double amplitudePointMinimumTime = 0.0;
    double amplitudePointMaximumTime = 0.0;
    bool amplitudeDragUsesLane = false;
    float amplitudeLaneMaxPercent = 200.0f;
    juce::Point<float> marqueeStart;
    juce::Point<float> marqueeCurrent;
    // Pixels per tick while a marquee is held against an edge, signed by
    // direction; 0 when the pointer is back inside and nothing should move.
    int marqueeScrollStep = 0;
    void updateMarqueeAutoScroll(juce::Point<float> position);
    void timerCallback() override;
    bool marqueeAddsToSelection = false;
    bool marqueeShowsContextMenu = false;
};
}
