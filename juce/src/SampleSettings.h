#pragma once

#include "ProjectModel.h"
#include "PronunciationTiming.h"
#include <juce_core/juce_core.h>
#include <vector>

namespace hachi
{
struct SampleRegionSetting
{
    juce::String name;
    double regionStartSeconds = 0.0;
    double regionEndSeconds = 0.5;
    double alignmentSeconds = 0.0;
    double fixedDurationSeconds = 0.0;
    double relativePitchCents = 0.0;
    bool melodyneData = false;
    double melodynePitchCenterCents = 0.0;
    double melodyneOriginalPitchCenterCents = 0.0;
    double melodynePitchDrift = 1.0;
    double melodynePitchModulation = 1.0;
    double melodyneTransitionSeconds = 0.0;
    double melodyneFormantCents = 0.0;
    double melodyneAmplitude = 1.0;
    double melodyneSibilantBalance = 0.0;
    double melodyneAttackSeconds = 0.0;
    double melodyneDecayElongation = 0.0;
    // UTAU's sixth oto.ini value.  Kept separately from alignment/preutterance
    // so the wavtool-style mixer can reproduce the intended crossfade.
    double overlapSeconds = 0.0;
    // v2 native metadata. Appended after legacy fields so old aggregate
    // initialisers and old CSV columns remain source-compatible.
    int hjmVersion = 2;
    NativeSegmentRole role = NativeSegmentRole::unknown;
    juce::String provenance = "estimated";
    float confidence = 0.0f;
    std::vector<NativeSegment> segments;
    std::vector<AmplitudeEnvelopePoint> amplitudeEnvelope;
    bool nativeUnpitched = false;
    // Folder-level otto.ini identities. Empty only for legacy/derived rows;
    // native material transactions assign stable IDs before publication.
    juce::String materialAudioId;
    juce::String materialRegionId;
    juce::StringArray additionalAliases;
    juce::String connectedNextRegionId;
    PronunciationKind pronunciationKind = PronunciationKind::normal;
    // Independent C/P editing by default. When linked, C-P is derived from the
    // existing marker values, not duplicated in another persisted offset.
    bool simplifiedConsonant = false;
    // Only explicitly named manual vowel cuts are authoring regions. Pitch
    // analysis/project-note epochs must never populate this intent implicitly.
    bool vowelSplitsExplicit = false;
};

struct VoicebankOtoEntry
{
    juce::File otoFile;
    juce::File audioFile;
    juce::String sourceName;
    juce::String alias;
    double offsetMs = 0.0;
    double consonantMs = 0.0;
    double cutoffMs = 0.0;
    double preutteranceMs = 0.0;
    double overlapMs = 0.0;
    // -1 while this entry has no row in the oto file yet: a sample sitting
    // in the folder that nothing has described.  Saving it appends a row.
    int lineIndex = -1;
    // Jie-oto extension, stored beside oto.ini in oto4.ini.  The three inner
    // boundaries are milliseconds relative to offsetMs and split the sample
    // into onset / glide / nucleus / coda, which is what the WCSNDM engine
    // needs to give each region its own output length.  hasJieOto is false
    // until the voicebank has been extended, and the entry then behaves
    // exactly like a classic two-region oto entry.
    bool hasJieOto = false;
    double jieOnsetMs = 0.0;    // onset | glide
    double jieGlideMs = 0.0;    // glide | nucleus
    double jieNucleusMs = 0.0;  // nucleus | coda
    // 谋-oto (otomou.ini): one letter per region saying what it is --
    // 'C' consonant, 'V' vowel, 'S' silence.  Empty when the voicebank has
    // no annotation, which is every voicebank until someone writes one, and
    // is what UTAU and 界•UTAU always send: those two modes have no notion
    // of the annotation and the engine must not be told about one.
    juce::String mouClasses;
    juce::String sourceEncoding, sourceFingerprint;
};

class SampleSettings final
{
public:
    // Legacy read-only lookup; native writes must use annotationFor().
    static juce::File sidecarFor(const juce::File& audio);
    // One manifest per registered material root; nested audio uses relative paths.
    static juce::File annotationFor(const juce::File& audio);
    static void setMaterialRoots(const juce::StringArray& roots);
    static juce::StringArray materialRoots();
    static bool hasNativeAnnotation(const juce::File& audio, juce::String* error = nullptr);
    // Copy only this audio's annotations, never an entire shared manifest.
    static bool copyAnnotations(const juce::File& source, const juce::File& destination,
                                juce::String& error);
    // Convert one legacy or v2 row to the native, arbitrarily segmented form
    // consumed by the editor and renderers.  Returned times are local to the
    // row's source region.
    [[nodiscard]] static std::vector<NativeSegment> nativeSegmentsFor(
        const SampleRegionSetting& row);
    // Native variant ambiguity returns empty with error, never derived rows.
    // Callers needing diagnostics must provide error; an explicit ID cannot derive.
    static std::vector<SampleRegionSetting> loadOrDerive(const juce::File& audio,
                                                         const ProjectData& project,
                                                         juce::String* error = nullptr,
                                                         bool explicitLegacyMigration = false,
                                                          const juce::String& audioId = {});
    static bool save(const juce::File& audio, const std::vector<SampleRegionSetting>& rows,
                     juce::String& error, const juce::String& audioId = {});
    // Convert the already-parsed Melodyne project into native HJM annotations
    // and bind the resulting segments back to its native notes.  This is the
    // single conversion service used by GUI and headless imports.
    static bool convertMelodyneProject(ProjectData& project, juce::StringArray& warnings);
    static bool importOto(const juce::File& oto, const juce::File& audio,
                          double audioDuration, std::vector<SampleRegionSetting>& rows,
                          juce::String& error);
    static bool exportOto(const juce::File& oto, const juce::File& audio,
                          const std::vector<SampleRegionSetting>& rows, double audioDuration,
                          juce::String& error);
    static bool importVoicebank(const juce::File& root, juce::StringArray& audioFiles,
                                int& sidecarsWritten, int& regionsWritten,
                                juce::StringArray& warnings);
    // A note's own oto from what the editor produced, and the entry the editor
    // opens with for a note that already has one.  Neither touches a file.
    // withRegions says whether the track reads regions at all: an entry opened
    // in 界 or 谋 is given its boundaries by the editor even if the voicebank
    // had none written.
    static backend::UtauOtoOverride noteOtoFromEntry(const VoicebankOtoEntry& entry,
                                                     bool withRegions)
    {
        backend::UtauOtoOverride oto;
        oto.enabled = true;
        oto.offsetMs = entry.offsetMs;
        oto.consonantMs = entry.consonantMs;
        oto.cutoffMs = entry.cutoffMs;
        oto.preutteranceMs = entry.preutteranceMs;
        oto.overlapMs = entry.overlapMs;
        oto.hasRegions = withRegions || entry.hasJieOto;
        oto.onsetMs = entry.jieOnsetMs;
        oto.glideMs = entry.jieGlideMs;
        oto.nucleusMs = entry.jieNucleusMs;
        oto.classes = entry.mouClasses;
        return oto;
    }
    static VoicebankOtoEntry entryWithNoteOto(VoicebankOtoEntry entry,
                                              const backend::UtauOtoOverride& oto)
    {
        if (!oto.enabled) return entry;
        entry.offsetMs = oto.offsetMs;
        entry.consonantMs = oto.consonantMs;
        entry.cutoffMs = oto.cutoffMs;
        entry.preutteranceMs = oto.preutteranceMs;
        entry.overlapMs = oto.overlapMs;
        entry.hasJieOto = oto.hasRegions;
        entry.jieOnsetMs = oto.onsetMs;
        entry.jieGlideMs = oto.glideMs;
        entry.jieNucleusMs = oto.nucleusMs;
        entry.mouClasses = oto.classes;
        return entry;
    }
    // mouMode lays 谋•OTO over the entries as well.  It is off for UTAU and
    // 界•UTAU, which have no annotation: with it off, an otomou.ini sitting
    // in the folder is never even opened.
    static std::vector<VoicebankOtoEntry> loadVoicebankOto(
        const juce::File& root, juce::StringArray& warnings,
        bool jieMode = false, bool mouMode = false,
        const juce::File& onlyOtoFile = {});
    // Read only the governing OTO files for this recording, nearest folder first.
    static std::vector<VoicebankOtoEntry> loadAudioOto(const juce::File& audio,
        const juce::File& root, UtauMode mode, juce::StringArray& warnings);
    // Materialise authored OTO trim/regions as ordinary editable native notes.
    static std::optional<ClipData> nativeAudioFromOto(const VoicebankOtoEntry& entry,
        UtauMode mode, double sourceDurationSeconds);
    // Counts the writes made here to the files a voicebank is read from --
    // oto.ini, oto.jie.ini, oto4.ini, otomou.ini and the sample sidecars.  The
    // renderer keys its voicebank index on it, so what is written here is what
    // the next note is sung from, with no wait.
    [[nodiscard]] static std::uint64_t voicebankFilesRevision();
    static bool updateVoicebankOtoEntry(const VoicebankOtoEntry& original,
                                        const VoicebankOtoEntry& updated,
                                        juce::String& error);
    // Jie mode keeps its classic six oto fields in oto.jie.ini.  The file is
    // seeded from oto.ini on first use and is never allowed to rewrite the
    // original voicebank configuration.
    static juce::File jieClassicOtoFileFor(const juce::File& otoFile);
    static bool updateJieVoicebankOtoEntry(const VoicebankOtoEntry& original,
                                           const VoicebankOtoEntry& updated,
                                           juce::String& error);
    // Adds another alias/timing row that points at the same audio filename.
    // No WAV/FLAC data is copied.
    static bool duplicateVoicebankOtoEntry(const VoicebankOtoEntry& source,
                                           const juce::String& newAlias,
                                           bool jieMode, juce::String& error);
    // oto.ini's four-region companion.  The file is named oto4.ini on disk
    // because that is the name the WCSNDM engine looks for; the interface
    // calls it Jie-oto.
    // The name an entry answers to: its alias, or the file stem when the oto
    // line leaves the alias empty, which is how UTAU resolves it.
    static juce::String entryLookupName(const VoicebankOtoEntry& entry);
    // Index of the entry matching an alias, else the alphabetically nearest.
    // -1 when there are no entries.
    static int findEntryForAlias(const std::vector<VoicebankOtoEntry>& entries,
                                 const juce::String& alias);
    static juce::File jieOtoFileFor(const juce::File& otoFile);
    // Fills in the four-region fields of entries already loaded from oto.ini.
    static void mergeJieOto(std::vector<VoicebankOtoEntry>& entries);
    // Lay 谋•OTO over the entries: the per-region classes, and the boundaries
    // that go with them.  Only called in 谋•UTAU mode.
    static void mergeMouOto(std::vector<VoicebankOtoEntry>& entries);
    // How many regions a class string stands for: its length, or four when
    // there is none -- which is what an oto4 row has always had.
    static int mouRegionCount(const juce::String& classes);
    // The same annotation at a different count.  The letters that survive are
    // kept and anything new defaults to a vowel: the part that carries the
    // note is the safe guess, and the author says otherwise by typing.
    static juce::String mouClassesForCount(const juce::String& classes, int count);
    // Write one entry's row into otomou.ini, keeping its class string.  The
    // classes are the whole point of the file, so an edit that only moved a
    // boundary must not drop them.
    static bool updateMouOtoEntry(const VoicebankOtoEntry& original,
                                  const VoicebankOtoEntry& updated,
                                  juce::String& error);
    // Seed otomou.ini from what the voicebank already has, every entry
    // starting at CVVV -- which is exactly what an oto4 row has always meant,
    // so a freshly seeded file renders identically to 界.  Rows already there
    // are left alone.
    // merged counts rows removed because another row already had the same
    // wav and offset.  otomou.ini keys a row by those two, so a second row for
    // them is malformed -- seeding used to write one per alias, and an
    // annotation saved into such a file was written to one of them and read
    // back out of another.  Seeding repairs that, keeping the annotated row.
    static bool createMouOto(const juce::File& root, int& written, int& kept,
                             int& merged, juce::String& error);
    [[nodiscard]] static juce::File mouOtoFileFor(const juce::File& otoFile);
    // Seed oto4.ini from oto.ini so every alias already has a row to drag
    // apart.  Existing rows are kept, so this is safe to run again.
    static bool createJieOto(const juce::File& root, int& written, int& kept,
                             juce::String& error);
    struct JieFromMouResult { int written = 0, unchanged = 0, skipped = 0; };
    // Copy only explicit four-region Mou records matching the Jie sample and
    // offset. Preserve unrelated Jie rows and the source Mou annotations.
    static bool createJieOtoFromMou(const juce::File& root, JieFromMouResult& result,
                                    juce::String& error);
    static bool updateJieOtoEntry(const VoicebankOtoEntry& original,
                                  const VoicebankOtoEntry& updated,
                                  juce::String& error);
};
}
