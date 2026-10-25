#pragma once

#include <juce_core/juce_core.h>
#include <optional>
#include <vector>

namespace hachi::backend
{
// One note as a UST spells it.  Kept close to the file rather than to the
// application's own model, so the parser can be checked against a real UST
// without the conversion's opinions in the way.
struct UstNote
{
    // UST measures length in ticks, 480 to a quarter note, always.
    int sourceSectionIndex = -1;
    juce::String sourceSection;
    bool mode2 = true;
    std::optional<double> modulation, stpMs;
    double mode1StartMs = 0;
    std::vector<double> mode1Cents;
    int lengthTicks = 480;
    juce::String lyric;
    int noteNum = 60;
    juce::String flags;
    // A note may carry a tempo change from itself onwards.
    std::optional<double> tempo;
    // Milliseconds.  Absent means "use whatever the voicebank's oto says",
    // which is not the same as zero.
    std::optional<double> preutteranceMs;
    std::optional<double> overlapMs;
    std::optional<int> velocity;
    std::optional<int> intensity;

    // Mode-2 pitch bend.  PBS is the first point: an offset in milliseconds
    // from the note's start (normally negative, so the bend begins inside the
    // previous note) and a pitch.  Every pitch here is in tenths of a
    // semitone, relative to this note's own NoteNum -- a PBS pitch of -20 on
    // a note at 63 starts the bend at 61, which is how a portamento out of the
    // preceding note is written.
    bool hasPitchBend = false;
    double pitchStartMs = 0.0;
    double pitchStartTenths = 0.0;
    std::vector<double> widthsMs;      // PBW: the gap to each following point
    std::vector<double> pitchTenths;   // PBY: their pitches, short by the last
    juce::StringArray shapes;          // PBM: one per segment, "" s r or j

    // The amplitude envelope, as "Envelope=p1,p2,p3,v1,v2,v3,v4[,%,p4[,p5,v5]]".
    //
    // Times are milliseconds and volumes are percentages, 100 being the note's
    // own level.  The shape runs from silence at the beginning of the rendered
    // output -- which starts one preutterance before the note itself -- rises
    // over p1 then p2, holds, and comes back down over p4 then p3 measured
    // backwards from the end:
    //
    //     0 ... p1 ... p1+p2 .......... end-p3-p4 ... end-p3 ... end
    //     0      v1      v2                 v3          v4        0
    //
    // p1 usually equals the note's overlap, because that first ramp is the
    // crossfade with the note before it, and p4 usually equals the next note's
    // overlap for the same reason.  p5/v5 add one more point in the middle and
    // are rare; some files write p5 with no v5, which places nothing.
    // VBR: the seven numbers UTAU writes a vibrato as -- how much of the
    // note's end it covers (%), one swing (ms), its depth (cents), the fade in
    // and out (% of the vibrato), the phase it starts at (% of a swing) and
    // how far the swing's centre is moved off the note's pitch (% of depth).
    // Files carry an eighth field, which UTAU does not use.
    bool hasVibrato = false;
    double vibratoLengthPercent = 0.0;
    double vibratoCycleMs = 0.0;
    double vibratoDepthCents = 0.0;
    double vibratoFadeInPercent = 0.0;
    double vibratoFadeOutPercent = 0.0;
    double vibratoPhasePercent = 0.0;
    double vibratoOffsetPercent = 0.0;

    bool hasEnvelope = false;
    double envelopeP1 = 0.0, envelopeP2 = 0.0, envelopeP3 = 0.0;
    double envelopeP4 = 0.0;
    double envelopeV1 = 0.0, envelopeV2 = 100.0, envelopeV3 = 100.0, envelopeV4 = 0.0;
    bool hasMiddlePoint = false;
    double envelopeP5 = 0.0, envelopeV5 = 100.0;

    [[nodiscard]] bool isRest() const;
    // Optional absolute song-quarter positions/durations for USTX. Legacy UST
    // stays sequential in 480 ticks/quarter and leaves both fields unset.
    std::optional<double> positionQuarterNotes, durationQuarterNotes;
};

struct UstTempoChange
{
    double quarterPosition = 0.0, bpm = 120.0;
};
struct UstPart
{
    juce::String name, voiceDirectory, globalFlags;
    std::vector<UstNote> notes;
};

struct UstProject
{
    juce::String sourceText, sourceEncoding, sourceBytes;
    bool sourceBom = false, mode2 = true;
    double tempo = 120.0;
    juce::String name;
    // As written, usually "%VOICE%<folder>".  The application cannot resolve
    // it -- %VOICE% is wherever that UTAU install keeps its voicebanks -- so
    // it is carried out for the person to act on rather than guessed at.
    juce::String voiceDirectory;
    juce::String globalFlags;
    std::vector<UstNote> notes;
    // Root notes represent the first voice track. Flattened additional tracks
    // retain absolute musical time; raw document/warnings preserve unsupported
    // source UI grouping instead of silently losing its semantics.
    juce::String sourceFormat = "ust";
    std::vector<UstTempoChange> tempoChanges;
    int numerator = 4, denominator = 4;
    std::vector<UstPart> additionalTracks;
};

class UstImporter final
{
public:
    // Text already decoded.  Pure, so a check can hand it a fixture.
    [[nodiscard]] static UstProject parse(const juce::String& text,
                                          juce::StringArray& warnings);

    // A UST is written in whatever code page the machine that saved it used;
    // UTAU itself reads them that way.  Newer tools write UTF-8.  Valid UTF-8
    // is taken as UTF-8.  Anything else is read in the local code page and in
    // Shift-JIS -- and on a Chinese or Japanese machine in GBK too -- and the
    // reading with the most kana wins: a Japanese file opened on a Chinese
    // machine, read in the local code page alone, turned every kana into a
    // rare hanzi.  With no kana either way, the local code page, as before.
    [[nodiscard]] static juce::String decode(const juce::MemoryBlock& bytes,
                                             juce::String& encodingUsed);
    // The same, as a machine whose code page is localCodePage would read it,
    // which decides the readings tried.  So a check can be every machine.
    [[nodiscard]] static juce::String decode(const juce::MemoryBlock& bytes,
                                             juce::String& encodingUsed,
                                             int localCodePage);

    [[nodiscard]] static std::optional<UstProject> read(const juce::File& file,
                                                        juce::String& error,
                                                        juce::StringArray& warnings, int encoding = 0);

    // Ticks to quarter notes.  One place, because getting it wrong scales the
    // whole song and looks like a tempo bug.
    [[nodiscard]] static constexpr double quarterNotes(int ticks)
    {
        return static_cast<double>(ticks) / 480.0;
    }
};
}
