#include "UstxImporter.h"
// Format semantics inspected against OpenUtau commit
// 33302b3d7cecf9ae830d4229639e06788f717b3c (MIT):
// https://github.com/openutau/OpenUtau/blob/33302b3d7cecf9ae830d4229639e06788f717b3c/OpenUtau.Core/Format/USTx.cs
// https://github.com/openutau/OpenUtau/blob/33302b3d7cecf9ae830d4229639e06788f717b3c/OpenUtau.Core/Ustx/UNote.cs
// https://github.com/openutau/OpenUtau/blob/33302b3d7cecf9ae830d4229639e06788f717b3c/OpenUtau.Core/Util/Yaml.cs
// The public wiki's 0.6 version statement is older than that source's 0.10.
#include <yaml-cpp/yaml.h>
#include <yaml-cpp/eventhandler.h>
#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

namespace hachi::backend
{
namespace
{
constexpr size_t maxBytes = 16u * 1024u * 1024u;
constexpr int maxTick = 1000000000;

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

// Reject overlong forms, surrogate codepoints, truncated sequences and embedded NUL.
bool validUtf8(std::string_view s)
{
    for (size_t i = 0; i < s.size();)
    {
        const auto c = static_cast<unsigned char>(s[i++]);
        if (c == 0) return false;
        if (c < 0x80) continue;
        int count = 0; unsigned code = 0, minimum = 0;
        if (c >= 0xc2 && c <= 0xdf) { count = 1; code = c & 31; minimum = 0x80; }
        else if (c >= 0xe0 && c <= 0xef) { count = 2; code = c & 15; minimum = 0x800; }
        else if (c >= 0xf0 && c <= 0xf4) { count = 3; code = c & 7; minimum = 0x10000; }
        else return false;
        if (i + static_cast<size_t>(count) > s.size()) return false;
        while (count-- > 0)
        {
            const auto next = static_cast<unsigned char>(s[i++]);
            if ((next & 0xc0) != 0x80) return false;
            code = (code << 6) | (next & 63);
        }
        if (code < minimum || code > 0x10ffff || (code >= 0xd800 && code <= 0xdfff)) return false;
    }
    return true;
}

// This is a yaml-cpp SAX validation pass, not a second YAML implementation.
// Limit depth BEFORE constructing a node graph; aliases cannot create cycles or
// exponential expansion. OpenUtau's serializer disables aliases itself.
class Guard final : public YAML::EventHandler
{
    struct Frame { bool map = false, key = true; std::set<std::string> keys; };
    std::vector<Frame> stack;
    size_t events = 0;
    void event() { if (++events > 1000000) fail("USTX exceeds 1,000,000 YAML events"); }
    void value(bool scalar, const std::string& text = {})
    {
        event();
        if (stack.empty() || !stack.back().map) return;
        auto& f = stack.back();
        if (f.key)
        {
            if (!scalar) fail("USTX mapping keys must be scalar strings");
            if (!f.keys.insert(text).second) fail("Duplicate USTX mapping key: " + text);
        }
        f.key = !f.key;
    }
    void tag(const std::string& t)
    {
        if (t.empty() || t == "?" || t == "!") return;
        static const std::set<std::string> allowed {"tag:yaml.org,2002:str", "tag:yaml.org,2002:int",
            "tag:yaml.org,2002:float", "tag:yaml.org,2002:bool", "tag:yaml.org,2002:null",
            "tag:yaml.org,2002:map", "tag:yaml.org,2002:seq"};
        if (allowed.count(t) == 0) fail("Custom YAML tags are not accepted in USTX");
    }
    void begin(bool map, const std::string& t)
    {
        tag(t); value(false);
        if (stack.size() >= 64) fail("USTX exceeds YAML nesting depth 64");
        stack.push_back({map, true, {}});
    }
public:
    void OnDocumentStart(const YAML::Mark&) override { event(); }
    void OnDocumentEnd() override { event(); }
    void OnNull(const YAML::Mark&, YAML::anchor_t) override { value(false); }
    void OnAlias(const YAML::Mark&, YAML::anchor_t) override { fail("YAML aliases are not accepted in USTX"); }
    void OnScalar(const YAML::Mark&, const std::string& t, YAML::anchor_t, const std::string& v) override
    { tag(t); if (!validUtf8(v)) fail("Invalid Unicode or NUL in YAML scalar"); value(true, v); }
    void OnSequenceStart(const YAML::Mark&, const std::string& t, YAML::anchor_t, YAML::EmitterStyle::value) override { begin(false, t); }
    void OnSequenceEnd() override { event(); stack.pop_back(); }
    void OnMapStart(const YAML::Mark&, const std::string& t, YAML::anchor_t, YAML::EmitterStyle::value) override { begin(true, t); }
    void OnMapEnd() override { event(); stack.pop_back(); }
};

juce::String str(const std::string& s) { return juce::String::fromUTF8(s.data(), static_cast<int>(s.size())); }
void map(const YAML::Node& n, const std::string& path)
{ if (!n.IsMap()) fail(path + " must be a mapping"); }
YAML::Node sequence(const YAML::Node& n, const std::string& path, size_t limit)
{
    if (!n || n.IsNull()) return YAML::Node(YAML::NodeType::Sequence);
    if (!n.IsSequence()) fail(path + " must be a sequence");
    if (n.size() > limit) fail(path + " exceeds item limit " + std::to_string(limit));
    return n;
}
std::string scalar(const YAML::Node& n, const std::string& path)
{
    if (!n || !n.IsScalar()) fail(path + " must be a scalar");
    return n.Scalar();
}
juce::String text(const YAML::Node& n, const std::string& path, const char* fallback = "")
{ return !n || n.IsNull() ? juce::String(fallback) : str(scalar(n, path)); }
int integer(const YAML::Node& n, const std::string& path, int lo, int hi)
{
    auto s = scalar(n, path); int v = 0;
    const auto result = std::from_chars(s.data(), s.data() + s.size(), v);
    if (result.ec != std::errc() || result.ptr != s.data() + s.size() || v < lo || v > hi)
        fail(path + " must be a decimal integer in [" + std::to_string(lo) + "," + std::to_string(hi) + "]");
    return v;
}
int intOr(const YAML::Node& n, const std::string& path, int fallback, int lo, int hi)
{ return !n || n.IsNull() ? fallback : integer(n, path, lo, hi); }
double number(const YAML::Node& n, const std::string& path, double lo = -1.e12, double hi = 1.e12)
{
    auto s = scalar(n, path); double v = 0;
    const auto result = std::from_chars(s.data(), s.data() + s.size(), v, std::chars_format::general);
    if (result.ec != std::errc() || result.ptr != s.data() + s.size() || !std::isfinite(v) || v < lo || v > hi)
        fail(path + " must be a finite number within supported bounds");
    return v;
}
double numOr(const YAML::Node& n, const std::string& path, double fallback, double lo = -1.e12, double hi = 1.e12)
{ return !n || n.IsNull() ? fallback : number(n, path, lo, hi); }
bool boolean(const YAML::Node& n, const std::string& path, bool fallback)
{
    if (!n || n.IsNull()) return fallback;
    auto s = scalar(n, path);
    if (s == "true" || s == "True" || s == "TRUE") return true;
    if (s == "false" || s == "False" || s == "FALSE") return false;
    fail(path + " must be true or false");
}
void warn(juce::StringArray& warnings, const juce::String& message)
{
    if (warnings.size() < 1000) warnings.addIfNotAlreadyThere(message);
    else if (warnings.size() == 1000) warnings.add("Further USTX warnings omitted (1,000-warning limit); original sourceText retained");
}
void unknown(const YAML::Node& n, std::initializer_list<const char*> known,
             const std::string& path, juce::StringArray& warnings)
{
    for (const auto& field : n)
    {
        const auto key = field.first.Scalar();
        if (std::none_of(known.begin(), known.end(), [&](const char* k) { return key == k; }))
            warn(warnings, str(path + "." + key + ": unsupported field retained in sourceText; not applied to rendering"));
    }
}
void preserved(juce::StringArray& warnings, const std::string& path)
{ warn(warnings, str(path + ": retained as source data; rendering equivalence is not guaranteed")); }

struct NativeNote
{
    UstNote note;
    bool snapFirst = true;
    int tuning = 0;
};
NativeNote parseNote(const YAML::Node& n, int partPosition, const std::string& path,
                     juce::StringArray& warnings)
{
    map(n, path);
    NativeNote result;
    auto& note = result.note;
    const int position = integer(n["position"], path + ".position", -maxTick, maxTick);
    note.lengthTicks = integer(n["duration"], path + ".duration", 1, maxTick);
    if (static_cast<int64_t>(partPosition) + position < -maxTick
        || static_cast<int64_t>(partPosition) + position + note.lengthTicks > maxTick)
        fail(path + " endpoint exceeds 1,000,000,000 ticks");
    note.positionQuarterNotes = (static_cast<double>(partPosition) + position) / 480.0;
    note.durationQuarterNotes = note.lengthTicks / 480.0;
    note.noteNum = integer(n["tone"], path + ".tone", 0, 127);
    note.lyric = text(n["lyric"], path + ".lyric", "a");
    note.sourceSection = str(path);
    result.tuning = intOr(n["tuning"], path + ".tuning", 0, -12000, 12000);
    if (result.tuning != 0) preserved(warnings, path + ".tuning (cents)");
    if (n["phonemizer"] && !n["phonemizer"].IsNull()) preserved(warnings, path + ".phonemizer");

    const auto pitch = n["pitch"];
    if (pitch && !pitch.IsNull())
    {
        map(pitch, path + ".pitch");
        result.snapFirst = boolean(pitch["snap_first"], path + ".pitch.snap_first", true);
        const auto data = sequence(pitch["data"], path + ".pitch.data", 100000);
        double previousX = 0;
        bool supported = true;
        for (size_t i = 0; i < data.size(); ++i)
        {
            const auto p = data[i]; map(p, path + ".pitch.data[]");
            const auto x = number(p["x"], path + ".pitch.x", -1.e8, 1.e8);
            const auto y = number(p["y"], path + ".pitch.y", -12000, 12000);
            const auto shape = text(p["shape"], path + ".pitch.shape", "io");
            if (i == 0) { note.pitchStartMs = x; note.pitchStartTenths = y; }
            else
            {
                if (x <= previousX) fail(path + ".pitch.data x values must strictly increase");
                note.widthsMs.push_back(x - previousX);
                note.pitchTenths.push_back(y);
            }
            // Shape belongs to the segment LEAVING the upstream point.
            if (i + 1 < data.size())
            {
                if (shape == "io") note.shapes.add("");
                else if (shape == "l") note.shapes.add("s");
                else if (shape == "i") note.shapes.add("j");
                else if (shape == "o") note.shapes.add("r");
                else { supported = false; note.shapes.add(shape); }
            }
            previousX = x;
            unknown(p, {"x", "y", "shape"}, path + ".pitch.data[]", warnings);
        }
        note.hasPitchBend = supported && !data.IsNull() && data.size() >= 2;
        if (!supported) preserved(warnings, path + ".pitch spline/unknown shape (not substituted with another shape)");
        if (data.size() == 1) preserved(warnings, path + ".pitch single-point bend");
        unknown(pitch, {"data", "snap_first"}, path + ".pitch", warnings);
    }
    const auto vibrato = n["vibrato"];
    if (vibrato && !vibrato.IsNull())
    {
        map(vibrato, path + ".vibrato");
        note.vibratoLengthPercent = numOr(vibrato["length"], path + ".vibrato.length", 0, 0, 100);
        note.vibratoCycleMs = numOr(vibrato["period"], path + ".vibrato.period", 175, 5, 500);
        note.vibratoDepthCents = numOr(vibrato["depth"], path + ".vibrato.depth", 25, 0, 200);
        note.vibratoFadeInPercent = numOr(vibrato["in"], path + ".vibrato.in", 10, 0, 100);
        note.vibratoFadeOutPercent = numOr(vibrato["out"], path + ".vibrato.out", 10, 0, 100);
        note.vibratoPhasePercent = numOr(vibrato["shift"], path + ".vibrato.shift", 0, 0, 100);
        note.vibratoOffsetPercent = numOr(vibrato["drift"], path + ".vibrato.drift", 0, -100, 100);
        note.hasVibrato = note.vibratoLengthPercent > 0;
        const auto link = numOr(vibrato["vol_link"], path + ".vibrato.vol_link", 0, -100, 100);
        if (link != 0) preserved(warnings, path + ".vibrato.vol_link");
        unknown(vibrato, {"length", "period", "depth", "in", "out", "shift", "drift", "vol_link"}, path + ".vibrato", warnings);
    }
    const auto expressions = sequence(n["phoneme_expressions"], path + ".phoneme_expressions", 10000);
    for (const auto& e : expressions)
    {
        map(e, path + ".phoneme_expressions[]");
        if (e["index"] && !e["index"].IsNull()) integer(e["index"], path + ".expression.index", 0, 100000);
        scalar(e["abbr"], path + ".expression.abbr");
        number(e["value"], path + ".expression.value");
    }
    // Expressions are per phoneme, not per score note. Never collapse differing
    // indices into one velocity/volume, nor infer flags from an abbreviation.
    if (expressions.size() != 0) preserved(warnings, path + ".phoneme_expressions (per-phoneme values)");
    const auto overrides = sequence(n["phoneme_overrides"], path + ".phoneme_overrides", 10000);
    for (const auto& o : overrides)
    {
        map(o, path + ".phoneme_overrides[]");
        integer(o["index"], path + ".phoneme_override.index", 0, 100000);
        for (const auto* key : {"offset", "preutter_delta", "overlap_delta", "attack_time_delta", "release_time_delta"})
            if (o[key] && !o[key].IsNull()) number(o[key], path + ".phoneme_override." + key);
    }
    if (overrides.size() != 0) preserved(warnings, path + ".phoneme_overrides");
    unknown(n, {"position", "duration", "tone", "lyric", "pitch", "vibrato", "tuning", "phonemizer", "phoneme_expressions", "phoneme_overrides"}, path, warnings);
    return result;
}

void validateCurves(const YAML::Node& node, const std::string& path, juce::StringArray& warnings)
{
    const auto curves = sequence(node, path, 10000);
    for (const auto& c : curves)
    {
        map(c, path + "[]"); scalar(c["abbr"], path + ".abbr");
        const auto xs = sequence(c["xs"], path + ".xs", 1000000);
        const auto ys = sequence(c["ys"], path + ".ys", 1000000);
        if (xs.size() != ys.size()) fail(path + " xs/ys sizes differ");
        int previous = -maxTick - 1;
        for (size_t i = 0; i < xs.size(); ++i)
        {
            const auto x = integer(xs[i], path + ".xs[]", -maxTick, maxTick);
            if (x <= previous) fail(path + " xs must strictly increase");
            previous = x;
            integer(ys[i], path + ".ys[]", -1000000000, 1000000000);
        }
    }
    if (curves.size() != 0) preserved(warnings, path + " (original part-relative automation)");
}
} // namespace

std::optional<UstProject> UstxImporter::parse(std::string_view utf8, juce::String& error,
                                            juce::StringArray& warnings)
{
    error.clear();
    juce::StringArray localWarnings;
    try
    {
        if (utf8.empty() || utf8.size() > maxBytes) fail("USTX must contain 1 byte to 16 MiB");
        if (!validUtf8(utf8)) fail("USTX must be valid UTF-8 without embedded NUL");
        const std::string source(utf8);
        std::istringstream stream(source);
        YAML::Parser parser(stream);
        Guard guard;
        if (!parser.HandleNextDocument(guard)) fail("Empty USTX document");
        if (parser.HandleNextDocument(guard)) fail("USTX must contain exactly one YAML document");
        const auto root = YAML::Load(source);
        map(root, "project");
        if (!root["ustx_version"]) fail("Missing ustx_version: not an identified USTX project");
        const auto version = scalar(root["ustx_version"], "ustx_version");
        const auto dot = version.find('.');
        if (dot == std::string::npos || dot == 0 || dot + 1 == version.size()
            || version.find("..") != std::string::npos || version.back() == '.'
            || version.find_first_not_of("0123456789.") != std::string::npos)
            fail("ustx_version must be a dotted numeric version");
        if (version != "0.3" && version != "0.4" && version != "0.5" && version != "0.6"
            && version != "0.7" && version != "0.8" && version != "0.9" && version != "0.10")
            preserved(localWarnings, "ustx_version " + version + " (unverified schema version)");
        if (version == "0.3" || version == "0.4")
            preserved(localWarnings, "legacy USTX lyric/expression upgrade semantics (lyrics are not rewritten)");
        if (intOr(root["resolution"], "resolution", 480, 1, 1000000) != 480)
            fail("OpenUtau USTX resolution must be 480 ticks per quarter");

        UstProject project;
        project.sourceFormat = "ustx";
        project.sourceEncoding = "UTF-8";
        project.sourceBom = utf8.size() >= 3 && utf8.substr(0, 3) == "\xef\xbb\xbf";
        project.sourceText = str(source); // includes BOM/comments/unknown data, exactly once
        project.name = text(root["name"], "name", "USTX");
        project.tempo = numOr(root["bpm"], "bpm", 120, 1, 1000);
        project.numerator = intOr(root["beat_per_bar"], "beat_per_bar", 4, 1, 1024);
        project.denominator = intOr(root["beat_unit"], "beat_unit", 4, 1, 1024);
        const auto tempos = sequence(root["tempos"], "tempos", 10000);
        for (const auto& tempo : tempos)
        {
            map(tempo, "tempos[]");
            const auto tick = integer(tempo["position"], "tempos.position", -maxTick, maxTick);
            const auto bpm = number(tempo["bpm"], "tempos.bpm", 1, 1000);
            project.tempoChanges.push_back({tick / 480.0, bpm});
            unknown(tempo, {"position", "bpm"}, "tempos[]", localWarnings);
        }
        std::stable_sort(project.tempoChanges.begin(), project.tempoChanges.end(),
            [](const auto& a, const auto& b) { return a.quarterPosition < b.quarterPosition; });
        for (size_t i = 0; i < project.tempoChanges.size(); ++i)
        {
            const auto& t = project.tempoChanges[i];
            if (i != 0 && t.quarterPosition == project.tempoChanges[i - 1].quarterPosition)
                fail("Duplicate tempo position");
            if (t.quarterPosition <= 0) project.tempo = t.bpm;
        }
        const auto signatures = sequence(root["time_signatures"], "time_signatures", 10000);
        std::set<int> bars;
        int appliedBar = std::numeric_limits<int>::min();
        for (const auto& signature : signatures)
        {
            map(signature, "time_signatures[]");
            const auto bar = integer(signature["bar_position"], "time_signatures.bar_position", -maxTick, maxTick);
            const auto numerator = integer(signature["beat_per_bar"], "time_signatures.beat_per_bar", 1, 1024);
            const auto denominator = integer(signature["beat_unit"], "time_signatures.beat_unit", 1, 1024);
            if (!bars.insert(bar).second) fail("Duplicate time-signature bar position");
            if ((denominator & (denominator - 1)) != 0) fail("Time-signature beat_unit must be a power of two");
            if (bar <= 0 && bar >= appliedBar)
            {
                project.numerator = numerator; project.denominator = denominator; appliedBar = bar;
            }
            unknown(signature, {"bar_position", "beat_per_bar", "beat_unit"}, "time_signatures[]", localWarnings);
        }
        if (signatures.size() > 1 || (signatures.size() != 0 && appliedBar != 0))
            preserved(localWarnings, "time_signatures map (only meter at song origin is applied)");
        const auto descriptors = root["expressions"];
        if (descriptors && !descriptors.IsNull())
        {
            map(descriptors, "expressions");
            for (const auto& descriptor : descriptors)
            {
                map(descriptor.second, "expressions." + descriptor.first.Scalar());
                for (const auto* key : {"min", "max", "default_value", "_custom_default_value"})
                    if (descriptor.second[key] && !descriptor.second[key].IsNull())
                        number(descriptor.second[key], "expressions." + descriptor.first.Scalar() + "." + key);
            }
            if (descriptors.size() != 0) preserved(localWarnings, "expressions descriptors/defaults/flag mappings");
        }

        const auto tracks = sequence(root["tracks"], "tracks", 256);
        if (tracks.size() == 0) fail("USTX must define at least one track");
        std::vector<UstPart> parsedTracks(tracks.size());
        std::vector<bool> voiceTrack(tracks.size(), false);
        for (size_t t = 0; t < tracks.size(); ++t)
        {
            const auto track = tracks[t]; map(track, "tracks[]");
            auto& out = parsedTracks[t];
            const auto path = "tracks[" + std::to_string(t) + "]";
            out.name = text(track["track_name"], path + ".track_name", "USTX Track");
            // singer is an OpenUtau identifier, NEVER a host voiceDirectory.
            // Keep it in raw text, rather than binding a path or inferring type.
            if (track["singer"] && !track["singer"].IsNull()) preserved(localWarnings, path + ".singer (not bound to a host bank)");
            if (track["phonemizer"] && !track["phonemizer"].IsNull()) preserved(localWarnings, path + ".phonemizer");
            numOr(track["volume"], path + ".volume", 0, -1000, 1000);
            numOr(track["pan"], path + ".pan", 0, -1000, 1000);
            boolean(track["mute"], path + ".mute", false);
            boolean(track["solo"], path + ".solo", false);
            unknown(track, {"track_name", "singer", "phonemizer"}, path, localWarnings);
        }
        const auto parts = sequence(root["voice_parts"], "voice_parts", 10000);
        size_t totalNotes = 0;
        for (size_t p = 0; p < parts.size(); ++p)
        {
            const auto part = parts[p]; const auto path = "voice_parts[" + std::to_string(p) + "]";
            map(part, path);
            const auto trackNo = integer(part["track_no"], path + ".track_no", 0, static_cast<int>(tracks.size()) - 1);
            const auto position = intOr(part["position"], path + ".position", 0, -maxTick, maxTick);
            const auto duration = intOr(part["duration"], path + ".duration", 0, 0, maxTick);
            if (static_cast<int64_t>(position) + duration > maxTick) fail(path + " endpoint exceeds tick limit");
            voiceTrack[static_cast<size_t>(trackNo)] = true;
            const auto notes = sequence(part["notes"], path + ".notes", 100000);
            totalNotes += notes.size();
            if (totalNotes > 100000) fail("USTX exceeds 100,000 notes");
            std::vector<NativeNote> nativeNotes;
            nativeNotes.reserve(notes.size());
            for (size_t i = 0; i < notes.size(); ++i)
                nativeNotes.push_back(parseNote(notes[i], position, path + ".notes[" + std::to_string(i) + "]", localWarnings));
            std::stable_sort(nativeNotes.begin(), nativeNotes.end(), [](const auto& a, const auto& b)
                { return *a.note.positionQuarterNotes < *b.note.positionQuarterNotes; });
            for (size_t i = 0; i < nativeNotes.size(); ++i)
            {
                auto& current = nativeNotes[i];
                if (current.snapFirst && current.note.hasPitchBend)
                {
                    auto first = 0.0;
                    if (i > 0)
                    {
                        const auto& prev = nativeNotes[i - 1];
                        const auto end = *prev.note.positionQuarterNotes + *prev.note.durationQuarterNotes;
                        if (std::abs(end - *current.note.positionQuarterNotes) < 1.e-9)
                            first = (prev.note.noteNum - current.note.noteNum) * 10.0
                                  + (prev.tuning - current.tuning) / 10.0;
                    }
                    current.note.pitchStartTenths = first;
                }
                parsedTracks[static_cast<size_t>(trackNo)].notes.push_back(std::move(current.note));
            }
            validateCurves(part["curves"], path + ".curves", localWarnings);
            unknown(part, {"track_no", "position", "duration", "notes", "curves"}, path, localWarnings);
            preserved(localWarnings, path + " part boundaries/grouping (note absolute positions retained)");
        }
        // Merge only the source UI grouping, NEVER sequentialize notes. Empty
        // gaps and overlaps survive as independent absolute quarter positions.
        bool first = true;
        for (size_t i = 0; i < parsedTracks.size(); ++i)
        {
            if (!voiceTrack[i])
            {
                if (parts.size() != 0) preserved(localWarnings, "tracks[" + std::to_string(i) + "] without voice parts (not created as a score track)");
                continue;
            }
            auto& track = parsedTracks[i];
            std::stable_sort(track.notes.begin(), track.notes.end(), [](const auto& a, const auto& b)
                { return *a.positionQuarterNotes < *b.positionQuarterNotes; });
            if (first)
            {
                project.name = track.name;
                project.notes = std::move(track.notes);
                first = false;
            }
            else project.additionalTracks.push_back(std::move(track));
        }
        if (first) project.name = parsedTracks.front().name;
        const auto waves = sequence(root["wave_parts"], "wave_parts", 10000);
        if (waves.size() + parts.size() > 10000) fail("USTX exceeds 10,000 combined voice/wave parts");
        if (waves.size() != 0)
            preserved(localWarnings, "wave_parts (external paths are inert; audio is not read or imported)");
        unknown(root, {"ustx_version", "resolution", "name", "bpm", "beat_per_bar", "beat_unit", "tempos", "time_signatures", "tracks", "voice_parts", "wave_parts", "expressions"}, "project", localWarnings);
        warn(localWarnings, "USTX sourceText retains the complete UTF-8 document; unsupported controls are not a promise of OpenUtau render fidelity");
        warnings.addArray(localWarnings);
        return project;
    }
    catch (const std::exception& e)
    {
        error = "USTX parse failed: " + str(e.what());
        return std::nullopt;
    }
}

std::optional<UstProject> UstxImporter::read(const juce::File& file, juce::String& error,
                                           juce::StringArray& warnings)
{
    error.clear();
    // Read exactly this user-selected file. No YAML paths are ever resolved.
    auto input = file.createInputStream();
    if (!input || !input->openedOk()) { error = "Cannot open USTX file"; return std::nullopt; }
    const auto size = input->getTotalLength();
    if (size <= 0 || size > static_cast<juce::int64>(maxBytes))
    { error = "USTX file must contain 1 byte to 16 MiB"; return std::nullopt; }
    std::string bytes(static_cast<size_t>(size), '\0');
    if (input->read(bytes.data(), static_cast<int>(size)) != size || !input->isExhausted())
    { error = "USTX file changed or could not be completely read"; return std::nullopt; }
    return parse(bytes, error, warnings);
}
} // namespace hachi::backend
