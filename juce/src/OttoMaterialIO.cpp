#include "OttoMaterialIO.h"
#include <juce_cryptography/juce_cryptography.h>
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <limits>
#include <iomanip>
#include <locale>
#include <map>
#include <mutex>
#include <set>
#include <sstream>

namespace hachi
{
namespace
{
using S = juce::String;
using V = juce::var;
using D = OttoMaterialIO::Document;
using A = OttoMaterialIO::Audio;
bool fail(S& e, const S& s) { e = s; return false; }
S number(double x)
{
    std::ostringstream out; out.imbue(std::locale::classic());
    out << std::setprecision(17) << x; return S(out.str());
}
double millisecondsValue(double seconds)
{
    const auto ms = seconds * 1000.0;
    const auto integer = std::round(ms);
    return std::abs(ms - integer) <= 1.0e-9 ? integer : ms;
}
S milliseconds(double seconds) { return number(millisecondsValue(seconds)); }
double recoveredSeconds(double seconds) { return millisecondsValue(seconds) / 1000.0; }
bool numeric(const S& s, double& n)
{
    auto bytes = s.toStdString();
    if (bytes.empty()) return false;
    auto result = std::from_chars(bytes.data(), bytes.data() + bytes.size(), n);
    return result.ec == std::errc{} && result.ptr == bytes.data() + bytes.size() && std::isfinite(n);
}
bool idValid(const S& s)
{
    return s.isNotEmpty() && s.length() <= 128
        && s.containsOnly("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-");
}
S mint(char prefix) { return S::charToString(prefix) + juce::Uuid().toString().removeCharacters("-"); }
juce::StringArray split(const S& s, juce::juce_wchar delimiter)
{
    juce::StringArray a; int start = 0;
    for (int i = 0; i < s.length(); ++i)
        if (s[i] == delimiter) { a.add(s.substring(start, i)); start = i + 1; }
    a.add(s.substring(start)); return a;
}
S escape(const S& s)
{
    S out; const auto* p = reinterpret_cast<const unsigned char*>(s.toRawUTF8());
    const char* hex = "0123456789ABCDEF";
    for (; *p; ++p)
    {
        const auto c = *p;
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
            || (c >= '0' && c <= '9') || c == '/' || c == '.' || c == '_' || c == '-')
            out += S::charToString(c);
        else { out += '%'; out += S::charToString(hex[c >> 4]); out += S::charToString(hex[c & 15]); }
    }
    return out;
}
bool unescape(const S& s, S& out, S& error)
{
    std::string bytes;
    auto hex = [](juce::juce_wchar c) -> int {
        if (c >= '0' && c <= '9') return int(c - '0');
        if (c >= 'a' && c <= 'f') return int(c - 'a' + 10);
        if (c >= 'A' && c <= 'F') return int(c - 'A' + 10);
        return -1;
    };
    const auto raw = s.toStdString();
    for (size_t i = 0; i < raw.size(); ++i)
    {
        auto c = static_cast<unsigned char>(raw[i]);
        if (c == '%')
        {
            if (i + 2 >= raw.size() || hex(raw[i + 1]) < 0 || hex(raw[i + 2]) < 0)
                return fail(error, "Invalid percent escape");
            c = static_cast<unsigned char>((hex(raw[i + 1]) << 4) | hex(raw[i + 2])); i += 2;
        }
        else if (c < 32 || c == 127 || c == '|' || c == ';' || c == '[' || c == ']' || c == '=' || c == '\\')
            return fail(error, "Reserved character must be percent escaped");
        if (c == 0) return fail(error, "NUL is not allowed");
        bytes.push_back(static_cast<char>(c));
    }
    if (!juce::CharPointer_UTF8::isValidString(bytes.data(), int(bytes.size())))
        return fail(error, "Invalid UTF-8 escape sequence");
    out = S::fromUTF8(bytes.data(), int(bytes.size())); return true;
}
bool pathValid(const S& path)
{
    if (path.isEmpty() || path.startsWithChar('/') || path.containsAnyOf("\\:")) return false;
    for (const auto& part : split(path, '/'))
        if (part.isEmpty() || part == "." || part == "..") return false;
    return !path.containsChar(0) && !path.containsAnyOf("\r\n");
}
char roleCode(NativeSegmentRole role)
{
    switch (role)
    {
        case NativeSegmentRole::consonant: return 'C'; case NativeSegmentRole::vowel: return 'V';
        case NativeSegmentRole::transition: return 'T'; case NativeSegmentRole::silence: return 'S';
        case NativeSegmentRole::breath: return 'B'; case NativeSegmentRole::noise: return 'N';
        case NativeSegmentRole::ending: return 'E'; default: return 'U';
    }
}
bool roleFrom(const S& s, NativeSegmentRole& role)
{
    const S codes("UCVTSBNE"); int i = codes.indexOf(s);
    if (s.length() != 1 || i < 0) return false;
    role = static_cast<NativeSegmentRole>(i); return true;
}
#define DIRECT_NUMBERS(X) X(regionStartSeconds) X(regionEndSeconds) X(fixedDurationSeconds) X(overlapSeconds)
#define CORE_NUMBERS(X) DIRECT_NUMBERS(X) X(alignmentSeconds)
#define ROW_NUMBERS(X) \
 X(relativePitchCents) X(melodynePitchCenterCents) X(melodyneOriginalPitchCenterCents) \
 X(melodynePitchDrift) X(melodynePitchModulation) X(melodyneTransitionSeconds) \
 X(melodyneFormantCents) X(melodyneAmplitude) X(melodyneSibilantBalance) \
 X(melodyneAttackSeconds) X(melodyneDecayElongation)
#define SEG_NUMBERS(X) X(sourceStartSeconds) X(sourceEndSeconds) X(alignmentSeconds) X(overlapSeconds) X(stretchWeight)
// JSON's default decimal-place formatting can round tiny model values to
// zero. Use round-trip significant digits for every metadata double.
S exactJSON(const V& v)
{
    if (v.isDouble()) return number(double(v));
    if (auto* d = v.getDynamicObject())
    {
        S out("{"); bool first = true;
        for (const auto& p : d->getProperties())
        {
            if (!first) out += ","; first = false;
            out += juce::JSON::toString(V(p.name.toString()), true) + ":" + exactJSON(p.value);
        }
        return out + "}";
    }
    if (auto* a = v.getArray())
    {
        S out("["); bool first = true;
        for (const auto& item : *a) { if (!first) out += ","; first = false; out += exactJSON(item); }
        return out + "]";
    }
    return juce::JSON::toString(v, true);
}
V object() { return V(new juce::DynamicObject()); }
void put(V& o, const char* key, const V& v) { o.getDynamicObject()->setProperty(key, v); }
V segmentJSON(const NativeSegment& s)
{
    auto o = object();
    put(o, "id", s.id); put(o, "alias", s.alias); put(o, "role", int(s.role));
    put(o, "provenance", s.provenance); put(o, "confidence", double(s.confidence));
    put(o, "stretchable", s.stretchable);
#define ADD(n) put(o, #n, s.n);
    SEG_NUMBERS(ADD)
#undef ADD
    return o;
}
NativeSegment compactSegment(const SampleRegionSetting& row, size_t index,
                             NativeSegmentRole role, double start, double end)
{
    NativeSegment s;
    s.id = "segment_" + S(int(index + 1));
    s.alias = row.name.trim().isEmpty() ? S("-") : row.name.trim();
    s.role = role; s.sourceStartSeconds = start; s.sourceEndSeconds = end;
    s.provenance = row.provenance.isEmpty() ? S("estimated") : row.provenance;
    s.confidence = row.confidence; s.alignmentSeconds = row.alignmentSeconds;
    s.overlapSeconds = row.overlapSeconds; s.stretchable = role != NativeSegmentRole::consonant;
    return s;
}
void restoreCompactSegmentDefaults(SampleRegionSetting& row)
{
    double start = row.regionStartSeconds;
    for (size_t i = 0; i < row.segments.size(); ++i)
    {
        const auto end = row.segments[i].sourceEndSeconds;
        row.segments[i] = compactSegment(row, i, row.segments[i].role, start, end);
        start = end;
    }
}
V residual(const SampleRegionSetting& r)
{
    auto o = object(); SampleRegionSetting defaults;
    if (!r.segments.empty()) defaults.role = r.segments.front().role;
#define EXACT(n) if (recoveredSeconds(r.n) != r.n) put(o, #n, r.n);
    DIRECT_NUMBERS(EXACT)
#undef EXACT
    if (recoveredSeconds(r.regionStartSeconds)
        + recoveredSeconds(r.alignmentSeconds - r.regionStartSeconds) != r.alignmentSeconds)
        put(o, "alignmentSeconds", r.alignmentSeconds);
#define ADD(n) if (r.n != defaults.n) put(o, #n, r.n);
    ROW_NUMBERS(ADD)
    ADD(melodyneData) ADD(hjmVersion) ADD(nativeUnpitched) ADD(provenance)
#undef ADD
    if (r.role != defaults.role) put(o, "role", int(r.role));
    if (r.confidence != defaults.confidence) put(o, "confidence", double(r.confidence));
    if (!r.segments.empty())
    {
        // Core carries role/end only. Emit full segment records only when those
        // defaults cannot exactly recover the original segment information.
        bool needed = false; double start = r.regionStartSeconds;
        for (const auto& s : r.segments)
        {
            const auto baseline = compactSegment(r, size_t(&s - r.segments.data()), s.role, start, s.sourceEndSeconds);
            if (recoveredSeconds(s.sourceEndSeconds) != s.sourceEndSeconds) needed = true;
            if (s.id != baseline.id || s.alias != baseline.alias || s.sourceStartSeconds != baseline.sourceStartSeconds
                || s.provenance != baseline.provenance || s.confidence != baseline.confidence
                || s.alignmentSeconds != baseline.alignmentSeconds || s.overlapSeconds != baseline.overlapSeconds
                || s.stretchable != baseline.stretchable || s.stretchWeight != baseline.stretchWeight) needed = true;
            start = s.sourceEndSeconds;
        }
        if (needed)
        {
            juce::Array<V> a; for (const auto& s : r.segments) a.add(segmentJSON(s));
            put(o, "segments", V(a));
        }
    }
    if (!r.amplitudeEnvelope.empty())
    {
        juce::Array<V> a;
        for (const auto& p : r.amplitudeEnvelope)
        {
            auto v = object(); put(v, "timeSeconds", p.timeSeconds); put(v, "gainDb", double(p.gainDb));
            put(v, "linearToNext", p.linearToNext); put(v, "nativeSeamAnchor", p.nativeSeamAnchor); a.add(v);
        }
        put(o, "amplitudeEnvelope", V(a));
    }
    return o;
}
bool keysOnly(const V& o, const std::set<S>& known, S& error)
{
    auto* d = o.getDynamicObject(); if (d == nullptr) return fail(error, "Metadata must be a JSON object");
    for (const auto& p : d->getProperties())
        if (!known.count(p.name.toString())) return fail(error, "Unknown metadata key: " + p.name.toString());
    return true;
}
bool getNum(const V& o, const char* key, double& n, S& e, bool required = false)
{
    if (!o.hasProperty(key)) return !required || fail(e, S("Missing metadata number: ") + key);
    auto v = o[key];
    if (!(v.isDouble() || v.isInt() || v.isInt64()) || !std::isfinite(double(v)))
        return fail(e, S("Invalid metadata number: ") + key);
    n = double(v); return true;
}
bool getBool(const V& o, const char* key, bool& n, S& e, bool required = false)
{
    if (!o.hasProperty(key)) return !required || fail(e, S("Missing metadata boolean: ") + key);
    if (!o[key].isBool()) return fail(e, S("Invalid metadata boolean: ") + key);
    n = bool(o[key]); return true;
}
bool getString(const V& o, const char* key, S& n, S& e, bool required = false)
{
    if (!o.hasProperty(key)) return !required || fail(e, S("Missing metadata string: ") + key);
    if (!o[key].isString()) return fail(e, S("Invalid metadata string: ") + key);
    n = o[key].toString(); return true;
}
bool getRole(const V& o, NativeSegmentRole& role, S& e)
{
    double n = int(role); if (!getNum(o, "role", n, e)) return false;
    if (n != std::floor(n) || n < 0 || n > 7) return fail(e, "Invalid metadata role");
    role = static_cast<NativeSegmentRole>(int(n)); return true;
}
bool applyResidual(const V& o, SampleRegionSetting& r, S& e)
{
    std::set<S> keys { "melodyneData", "hjmVersion", "role", "provenance", "confidence", "segments", "amplitudeEnvelope", "nativeUnpitched" };
#define KEY(n) keys.insert(#n);
    ROW_NUMBERS(KEY)
    CORE_NUMBERS(KEY)
#undef KEY
    if (!keysOnly(o, keys, e)) return false;
#define EXACT(n) { double exact = r.n; if (!getNum(o, #n, exact, e)) return false; if (std::abs(exact - r.n) > 1.0e-9) return fail(e, "Metadata contradicts core: " #n); r.n = exact; }
    CORE_NUMBERS(EXACT)
#undef EXACT
#define READ(n) if (!getNum(o, #n, r.n, e)) return false;
    ROW_NUMBERS(READ)
#undef READ
    double version = r.hjmVersion, confidence = r.confidence;
    if (!getBool(o, "melodyneData", r.melodyneData, e) || !getBool(o, "nativeUnpitched", r.nativeUnpitched, e)
        || !getString(o, "provenance", r.provenance, e) || !getRole(o, r.role, e)
        || !getNum(o, "hjmVersion", version, e) || !getNum(o, "confidence", confidence, e)) return false;
    if (version < 0 || version > 2147483647.0 || version != std::floor(version)) return fail(e, "Invalid HJM version");
    r.hjmVersion = int(version); r.confidence = float(confidence);
    restoreCompactSegmentDefaults(r);
    if (o.hasProperty("segments"))
    {
        const auto* a = o["segments"].getArray(); if (!a) return fail(e, "segments must be an array");
        std::vector<NativeSegment> segments;
        for (const auto& v : *a)
        {
            std::set<S> sk { "id", "alias", "role", "provenance", "confidence", "stretchable" };
#define KEY(n) sk.insert(#n);
            SEG_NUMBERS(KEY)
#undef KEY
            NativeSegment s; double c = 0;
            if (!keysOnly(v, sk, e) || !getString(v, "id", s.id, e, true) || !getString(v, "alias", s.alias, e, true)
                || !getString(v, "provenance", s.provenance, e, true) || !v.hasProperty("role") || !getRole(v, s.role, e)
                || !getNum(v, "confidence", c, e, true) || !getBool(v, "stretchable", s.stretchable, e, true))
                return fail(e, "Invalid segment metadata: " + e);
            s.confidence = float(c);
#define READ(n) if (!getNum(v, #n, s.n, e, true)) return false;
            SEG_NUMBERS(READ)
#undef READ
            segments.push_back(s);
        }
        // Metadata is an extension, not permission to contradict edited core.
        if (segments.size() != r.segments.size()) return fail(e, "Metadata segment count contradicts core");
        for (size_t i = 0; i < segments.size(); ++i)
            if (segments[i].role != r.segments[i].role || std::abs(segments[i].sourceEndSeconds - r.segments[i].sourceEndSeconds) > 1.0e-9)
                return fail(e, "Metadata segment end/role contradicts core");
        r.segments = std::move(segments);
    }
    if (o.hasProperty("amplitudeEnvelope"))
    {
        auto* a = o["amplitudeEnvelope"].getArray(); if (!a) return fail(e, "amplitudeEnvelope must be an array");
        r.amplitudeEnvelope.clear();
        for (const auto& v : *a)
        {
            AmplitudeEnvelopePoint p; double gain = 0;
            if (!keysOnly(v, { "timeSeconds", "gainDb", "linearToNext", "nativeSeamAnchor" }, e)
                || !getNum(v, "timeSeconds", p.timeSeconds, e, true) || !getNum(v, "gainDb", gain, e, true)
                || !getBool(v, "linearToNext", p.linearToNext, e, true) || !getBool(v, "nativeSeamAnchor", p.nativeSeamAnchor, e, true)) return false;
            p.gainDb = float(gain); r.amplitudeEnvelope.push_back(p);
        }
    }
    return true;
}
S rowText(const SampleRegionSetting& r)
{
    S aliases = escape(r.name); for (const auto& a : r.additionalAliases) aliases += ";" + escape(a);
    S line = r.materialRegionId + "=" + aliases + "|" + milliseconds(r.regionStartSeconds) + "," + milliseconds(r.regionEndSeconds)
        + "|" + milliseconds(r.fixedDurationSeconds) + "," + milliseconds(r.alignmentSeconds - r.regionStartSeconds) + "," + milliseconds(r.overlapSeconds) + "|";
    if (r.segments.empty()) line += "-";
    else for (size_t i = 0; i < r.segments.size(); ++i)
    {
        if (i) line += ",";
        line += S::charToString(roleCode(r.segments[i].role)) + ":" + milliseconds(r.segments[i].sourceEndSeconds);
    }
    if (r.connectedNextRegionId.isNotEmpty()) line += "|next:" + r.connectedNextRegionId;
    juce::StringArray flags;
    if (r.pronunciationKind != PronunciationKind::normal)
        flags.add(pronunciationKindName(r.pronunciationKind));
    else if (r.role == NativeSegmentRole::transition || r.role == NativeSegmentRole::breath)
        flags.add("normal"); // Explicit identity overrides a preserved legacy internal role.
    if (r.simplifiedConsonant) flags.add("simple-consonant");
    if (r.vowelSplitsExplicit) flags.add("vowel-splits");
    if (!flags.isEmpty()) line += "|flags:" + flags.joinIntoString(",");
    const auto meta = residual(r);
    if (meta.getDynamicObject()->getProperties().size() != 0)
    {
        const auto json = exactJSON(meta);
        line += "|meta:" + juce::Base64::toBase64(json.toRawUTF8(), json.getNumBytesAsUTF8());
    }
    return line;
}
// JUCE's JSON object parser accepts repeated property names. Reject them
// before parsing so an extension cannot silently overwrite migration data.
bool uniqueJSONKeys(const S& json, S& e)
{
    const auto bytes = json.toStdString(); std::vector<std::set<S>> objects;
    int depth = 0; bool started = false;
    if (!json.trim().startsWithChar('{')) return fail(e, "Metadata must be an object");
    for (size_t i = 0; i < bytes.size(); ++i)
    {
        const auto c = bytes[i];
        if (depth == 0 && started && c != ' ' && c != '\t' && c != '\r' && c != '\n')
            return fail(e, "Trailing metadata content");
        if (c == '{') started = true;
        if (c == '{') { objects.emplace_back(); if (++depth > 128) return fail(e, "Metadata nesting is too deep"); }
        else if (c == '[') { if (++depth > 128) return fail(e, "Metadata nesting is too deep"); }
        else if (c == '}') { if (objects.empty()) return fail(e, "Malformed metadata object"); objects.pop_back(); --depth; }
        else if (c == ']') --depth;
        else if (c == '"')
        {
            const auto start = i++;
            for (; i < bytes.size(); ++i)
            {
                if (bytes[i] == '\\') { ++i; continue; }
                if (bytes[i] == '"') break;
            }
            if (i >= bytes.size()) return fail(e, "Unterminated metadata string");
            auto next = i + 1;
            while (next < bytes.size() && (bytes[next] == ' ' || bytes[next] == '\t' || bytes[next] == '\r' || bytes[next] == '\n')) ++next;
            if (next < bytes.size() && bytes[next] == ':')
            {
                if (objects.empty()) return fail(e, "Metadata key outside object");
                V single; const auto literal = S::fromUTF8(bytes.data() + start, int(i - start + 1));
                const auto status = juce::JSON::parse("{" + literal + ":0}", single);
                if (status.failed() || !single.getDynamicObject()) return fail(e, "Invalid metadata key");
                const auto key = single.getDynamicObject()->getProperties().getName(0).toString();
                if (!objects.back().insert(key).second) return fail(e, "Duplicate metadata key: " + key);
            }
        }
    }
    return depth == 0 || fail(e, "Unbalanced metadata JSON");
}
bool parseRow(const S& key, const S& value, const S& audioId, SampleRegionSetting& r, S& e)
{
    r.materialAudioId = audioId; r.materialRegionId = key;
    const auto fields = split(value, '|');
    if (fields.size() < 4) return fail(e, "Pronunciation requires aliases|start,end|fixed,pre,overlap|segments");
    const auto names = split(fields[0], ';');
    if (!unescape(names[0], r.name, e)) return false;
    for (int i = 1; i < names.size(); ++i) { S a; if (!unescape(names[i], a, e)) return false; r.additionalAliases.add(a); }
    auto range = split(fields[1], ','); auto timing = split(fields[2], ',');
    if (range.size() != 2 || timing.size() != 3) return fail(e, "Invalid pronunciation range/timing arity");
    if (!numeric(range[0], r.regionStartSeconds) || !numeric(range[1], r.regionEndSeconds)
        || !numeric(timing[0], r.fixedDurationSeconds) || !numeric(timing[1], r.alignmentSeconds) || !numeric(timing[2], r.overlapSeconds))
        return fail(e, "Invalid/non-finite pronunciation number");
    r.regionStartSeconds /= 1000; r.regionEndSeconds /= 1000;
    r.fixedDurationSeconds /= 1000; r.alignmentSeconds = r.regionStartSeconds + r.alignmentSeconds / 1000; r.overlapSeconds /= 1000;
    if (fields[3] != "-")
    {
        double start = r.regionStartSeconds;
        for (const auto& text : split(fields[3], ','))
        {
            auto pair = split(text, ':'); NativeSegment s;
            if (pair.size() != 2 || !roleFrom(pair[0], s.role) || !numeric(pair[1], s.sourceEndSeconds))
                return fail(e, "Invalid compact segment (expected C/V/T/S/B/N/E/U:absoluteEndMs)");
            s.sourceStartSeconds = start; s.sourceEndSeconds /= 1000; start = s.sourceEndSeconds; r.segments.push_back(s);
        }
    }
    if (!r.segments.empty()) r.role = r.segments.front().role;
    restoreCompactSegmentDefaults(r);
    std::set<S> seen;
    bool explicitKind = false;
    for (int i = 4; i < fields.size(); ++i)
    {
        const auto colon = fields[i].indexOfChar(':');
        if (colon < 1) return fail(e, "Invalid optional pronunciation field");
        const auto tag = fields[i].substring(0, colon), data = fields[i].substring(colon + 1);
        if (!seen.insert(tag).second) return fail(e, "Duplicate optional field: " + tag);
        if (tag == "next") r.connectedNextRegionId = data;
        else if (tag == "flags")
        {
            std::set<S> flagNames;
            for (const auto& flag : split(data, ','))
            {
                if (!flagNames.insert(flag).second) return fail(e, "Duplicate pronunciation flag: " + flag);
                if (flag == "simple-consonant") r.simplifiedConsonant = true;
                else if (flag == "vowel-splits") r.vowelSplitsExplicit = true;
                else if (flag == "normal" || flag == "transition" || flag == "breath")
                {
                    if (explicitKind) return fail(e, "Conflicting pronunciation identities");
                    r.pronunciationKind = parsePronunciationKind(flag); explicitKind = true;
                }
                else return fail(e, "Unknown pronunciation flag: " + flag);
            }
        }
        else if (tag == "meta")
        {
            if (i != fields.size() - 1) return fail(e, "meta must be the final field");
            juce::MemoryOutputStream bytes;
            if (!juce::Base64::convertFromBase64(bytes, data) || bytes.getDataSize() == 0
                || juce::Base64::toBase64(bytes.getData(), bytes.getDataSize()) != data
                || std::memchr(bytes.getData(), 0, bytes.getDataSize()) != nullptr)
                return fail(e, "Invalid/noncanonical metadata base64 or NUL");
            if (!juce::CharPointer_UTF8::isValidString(static_cast<const char*>(bytes.getData()), int(bytes.getDataSize()))) return fail(e, "Invalid metadata UTF-8");
            const auto json = S::fromUTF8(static_cast<const char*>(bytes.getData()), int(bytes.getDataSize()));
            if (!uniqueJSONKeys(json, e)) return false;
            V meta; const auto result = juce::JSON::parse(json, meta);
            if (result.failed()) return fail(e, "Invalid metadata JSON: " + result.getErrorMessage());
            if (!applyResidual(meta, r, e)) return false;
        }
        else return fail(e, "Unknown optional field: " + tag);
    }
    // Old documents had only an internal role and optional transition flag.
    // Explicit new identity wins; C/V are never promoted to syllable types.
    if (!explicitKind && (r.role == NativeSegmentRole::transition || r.role == NativeSegmentRole::breath))
        r.pronunciationKind = r.role == NativeSegmentRole::breath ? PronunciationKind::breath : PronunciationKind::transition;
    return true;
}
bool readBytes(const juce::File& file, juce::MemoryBlock& bytes, bool& existed, S& e)
{
    existed = file.exists();
    if (!existed) { bytes.reset(); return true; }
    if (!file.existsAsFile()) return fail(e, "Manifest is not a regular file: " + file.getFullPathName());
    if (!file.loadFileAsData(bytes)) return fail(e, "Cannot read manifest: " + file.getFullPathName());
    return true;
}
bool relativeAudio(const juce::File& manifest, const juce::File& audio, S& path, S& e)
{
    const auto root = manifest.getParentDirectory();
    if (!audio.isAChildOf(root)) return fail(e, "Audio must be below the manifest folder");
    path = audio.getRelativePathFrom(root).replaceCharacter('\\', '/');
    if (!pathValid(path)) return fail(e, "Unsafe relative audio path");
    return true;
}
bool samePath(const S& a, const S& b)
{
#if JUCE_WINDOWS
    return a.equalsIgnoreCase(b);
#else
    return a == b;
#endif
}
S fingerprintToken(const D& d)
{
    return d.existed ? "v1:" + juce::SHA256(d.fingerprint).toHexString() : S("absent");
}
struct Observation { S token; };
std::mutex observationsMutex;
std::map<S, Observation> observations;
S observationKey(const juce::File& m, const juce::File& a)
{
    auto key = m.getFullPathName() + "\n" + a.getFullPathName();
#if JUCE_WINDOWS
    key = key.toLowerCase();
#endif
    return key;
}
void remember(const juce::File& m, const juce::File& a, const D& d)
{
    std::lock_guard<std::mutex> lock(observationsMutex);
    observations[observationKey(m, a)] = { fingerprintToken(d) };
    while (observations.size() > 256) observations.erase(observations.begin());
}
}

bool OttoMaterialIO::validate(const Document& d, S& e)
{
    e.clear(); std::set<S> ids;
    for (const auto& a : d.audio)
    {
        if (!idValid(a.id) || !ids.insert(a.id).second) return fail(e, "Invalid/duplicate audio identity: " + a.id);
        if (!pathValid(a.relativePath)) return fail(e, "Unsafe audio path: " + a.relativePath);
        std::map<S, const SampleRegionSetting*> rows;
        for (const auto& r : a.rows)
        {
            if (!idValid(r.materialRegionId) || rows.count(r.materialRegionId) != 0)
                return fail(e, "Invalid/duplicate pronunciation identity: " + r.materialRegionId);
            if (r.materialAudioId != a.id) return fail(e, "Pronunciation audio identity mismatch");
            rows[r.materialRegionId] = &r;
            std::set<S> aliases;
            auto names = r.additionalAliases; names.insert(0, r.name);
            for (const auto& name : names)
                if (name.isEmpty() || name.containsChar(0) || !aliases.insert(name).second)
                    return fail(e, "Empty/duplicate alias within pronunciation: " + name);
            const double length = r.regionEndSeconds - r.regionStartSeconds;
            if (!std::isfinite(r.regionStartSeconds) || !std::isfinite(r.regionEndSeconds) || r.regionStartSeconds < 0 || length <= 0
                || !std::isfinite(r.fixedDurationSeconds) || r.fixedDurationSeconds < 0 || r.fixedDurationSeconds > length + 1.0e-9
                || !std::isfinite(r.alignmentSeconds) || r.alignmentSeconds < r.regionStartSeconds - 1.0e-9 || r.alignmentSeconds > r.regionEndSeconds + 1.0e-9
                || !std::isfinite(r.overlapSeconds))
                return fail(e, "Invalid pronunciation range/timing: " + r.materialRegionId);
            if (int(r.pronunciationKind) < 0 || int(r.pronunciationKind) > int(PronunciationKind::breath))
                return fail(e, "Invalid pronunciation identity kind: " + r.materialRegionId);
            if (r.simplifiedConsonant
                && r.regionStartSeconds + r.fixedDurationSeconds < r.alignmentSeconds - 1.0e-9)
                return fail(e, "Simplified consonant must not precede preutterance: " + r.materialRegionId);
#define FINITE(n) if (!std::isfinite(r.n)) return fail(e, "Non-finite metadata: " #n);
            ROW_NUMBERS(FINITE)
#undef FINITE
            if (!std::isfinite(r.confidence) || r.confidence < 0 || r.confidence > 1 || int(r.role) < 0 || int(r.role) > 7 || r.hjmVersion < 0)
                return fail(e, "Invalid row metadata");
            double lastEnd = r.regionStartSeconds; std::set<S> segmentIds;
            for (const auto& s : r.segments)
            {
#define FINITE(n) if (!std::isfinite(s.n)) return fail(e, "Non-finite segment metadata: " #n);
                SEG_NUMBERS(FINITE)
#undef FINITE
                if (s.sourceStartSeconds < lastEnd - 1.0e-9 || s.sourceEndSeconds <= s.sourceStartSeconds
                    || s.sourceEndSeconds > r.regionEndSeconds + 1.0e-9 || s.stretchWeight < 0
                    || !std::isfinite(s.confidence) || s.confidence < 0 || s.confidence > 1 || int(s.role) < 0 || int(s.role) > 7
                    || (s.id.isNotEmpty() && !segmentIds.insert(s.id).second)) return fail(e, "Invalid source segment: " + r.materialRegionId);
                if (r.vowelSplitsExplicit && s.role == NativeSegmentRole::vowel && s.stretchable
                    && (s.alias.trim().isEmpty() || s.alias.trim() == "-"))
                    return fail(e, "Manual vowel subdivisions require explicit names: " + r.materialRegionId);
                lastEnd = s.sourceEndSeconds;
            }
            double previous = -std::numeric_limits<double>::infinity();
            for (const auto& p : r.amplitudeEnvelope)
            {
                if (!std::isfinite(p.timeSeconds) || !std::isfinite(p.gainDb) || p.timeSeconds < previous) return fail(e, "Invalid amplitude envelope");
                previous = p.timeSeconds;
            }
        }
        for (const auto& r : a.rows)
        {
            if (r.connectedNextRegionId.isEmpty()) continue;
            const auto found = rows.find(r.connectedNextRegionId);
            if (found == rows.end() || found->second == &r) return fail(e, "Connection must target another pronunciation in the same audio");
            if (std::abs(r.regionEndSeconds - found->second->regionStartSeconds) > connectionToleranceSeconds + 1.0e-12)
                return fail(e, "Connected source ranges are not continuous (1ms tolerance)");
            std::set<S> visited; auto* next = &r;
            while (next != nullptr)
            {
                if (!visited.insert(next->materialRegionId).second) return fail(e, "Connection cycle");
                auto it = rows.find(next->connectedNextRegionId); next = it == rows.end() ? nullptr : it->second;
            }
        }
    }
    return true;
}

bool OttoMaterialIO::decode(const S& text, Document& document, S& e)
{
    e.clear(); D d; A* current = nullptr; bool header = false, version = false, unit = false; int lineNo = 0;
    auto input = text; if (input.startsWithChar(0xfeff)) input = input.substring(1);
    for (auto line : juce::StringArray::fromLines(input))
    {
        ++lineNo; line = line.trim();
        if (line.isEmpty() || line.startsWithChar('#') || line.startsWithChar(';')) continue;
        auto bad = [&](const S& reason) { return fail(e, "Line " + S(lineNo) + ": " + reason); };
        if (line.startsWithChar('['))
        {
            if (!line.endsWithChar(']')) return bad("Malformed section");
            const auto section = line.substring(1, line.length() - 1);
            if (section == "otto")
            {
                if (header || !d.audio.empty()) return bad("Duplicate/misplaced otto header");
                header = true; current = nullptr; continue;
            }
            if (!header || !version || !unit) return bad("Missing complete [otto] header");
            const auto colon = section.indexOfChar(':'); if (colon < 1) return bad("Unknown section");
            A audio; audio.id = section.substring(0, colon);
            if (!unescape(section.substring(colon + 1), audio.relativePath, e)) return bad(e);
            d.audio.push_back(std::move(audio)); current = &d.audio.back(); continue;
        }
        const auto eq = line.indexOfChar('='); if (eq < 1) return bad("Expected key=value");
        const auto key = line.substring(0, eq).trim(), value = line.substring(eq + 1);
        if (current == nullptr)
        {
            if (!header) return bad("Missing [otto] header");
            if (key == "version") { if (version || value != "1") return bad("Duplicate/unsupported version (only 1 supported)"); version = true; }
            else if (key == "unit") { if (unit || value != "ms") return bad("Duplicate/unsupported unit"); unit = true; }
            else return bad("Unknown header key (writes refused)");
        }
        else
        {
            SampleRegionSetting r;
            if (!parseRow(key, value, current->id, r, e)) return bad(e);
            current->rows.push_back(std::move(r));
        }
    }
    if (!header || !version || !unit) return fail(e, "Missing [otto], version=1 or unit=ms");
    if (!validate(d, e)) return false;
    document = std::move(d); return true;
}
bool OttoMaterialIO::encode(const Document& d, S& text, S& e)
{
    if (!validate(d, e)) return false;
    S out("[otto]\nversion=1\nunit=ms\n");
    for (const auto& a : d.audio)
    {
        out += "\n[" + a.id + ":" + escape(a.relativePath) + "]\n";
        for (const auto& r : a.rows) out += rowText(r) + "\n";
    }
    D check; if (!decode(out, check, e)) return false;
    text = out; return true;
}
bool OttoMaterialIO::loadDocument(const juce::File& manifest, Document& document, S& e)
{
    e.clear(); juce::MemoryBlock bytes; bool existed = false;
    if (!readBytes(manifest, bytes, existed, e)) return false;
    if (!existed) return fail(e, "Manifest does not exist: " + manifest.getFullPathName());
    if (bytes.getSize() == 0) return fail(e, "Manifest is empty");
    if (bytes.getSize() > size_t(std::numeric_limits<int>::max())
        || (bytes.getSize() != 0 && std::memchr(bytes.getData(), 0, bytes.getSize()) != nullptr))
        return fail(e, "Manifest contains NUL bytes or exceeds supported size");
    if (!juce::CharPointer_UTF8::isValidString(static_cast<const char*>(bytes.getData()), int(bytes.getSize())))
        return fail(e, "Manifest is not valid UTF-8");
    D d;
    if (!decode(S::fromUTF8(static_cast<const char*>(bytes.getData()), int(bytes.getSize())), d, e)) return false;
    d.fingerprint = bytes; d.existed = true; document = std::move(d); return true;
}
bool OttoMaterialIO::saveDocument(const juce::File& manifest, Document& d, S& e)
{
    S text; if (!encode(d, text, e)) return false;
    auto lockPath = manifest.getFullPathName();
#if JUCE_WINDOWS
    lockPath = lockPath.toLowerCase();
#endif
    juce::InterProcessLock processLock("HachiOtto-" + S::toHexString(lockPath.hashCode64()));
    if (!processLock.enter(5000)) return fail(e, "Manifest is busy (process lock timeout)");
    struct Unlock { juce::InterProcessLock& lock; ~Unlock() { lock.exit(); } } unlock { processLock };
    juce::MemoryBlock before; bool existed;
    if (!readBytes(manifest, before, existed, e)) return false;
    if (existed != d.existed || before != d.fingerprint) return fail(e, "Manifest conflict: content changed; reload before saving");
    juce::TemporaryFile temp(manifest);
    {
        juce::FileOutputStream output(temp.getFile());
        if (!output.openedOk()) return fail(e, "Cannot create same-folder temporary manifest");
        if (!output.write(text.toRawUTF8(), text.getNumBytesAsUTF8())) return fail(e, "Cannot write temporary manifest");
        output.flush(); if (output.getStatus().failed()) return fail(e, "Cannot flush temporary manifest");
    }
    D verified; if (!loadDocument(temp.getFile(), verified, e)) return false;
    juce::MemoryBlock again; bool existsAgain;
    if (!readBytes(manifest, again, existsAgain, e)) return false;
    if (existed != existsAgain || before != again) return fail(e, "Manifest conflict during write; original not overwritten");
    if (!temp.overwriteTargetFileWithTemporary()) return fail(e, "Atomic manifest replacement failed");
    d.fingerprint = verified.fingerprint; d.existed = true; return true;
}
juce::File OttoMaterialIO::findManifest(const juce::File& audio)
{
    auto dir = audio.getParentDirectory();
    for (;;)
    {
        auto candidate = dir.getChildFile("otto.ini");
        if (candidate.exists()) return candidate;
        auto parent = dir.getParentDirectory(); if (parent == dir) return {};
        dir = parent;
    }
}
bool OttoMaterialIO::loadAudio(const juce::File& manifest, const juce::File& audio,
                                std::vector<SampleRegionSetting>& rows, S& e, S* loadedFingerprint,
                                const S& audioId)
{
    e.clear(); S path; if (!relativeAudio(manifest, audio, path, e)) return false;
    D d; if (!loadDocument(manifest, d, e)) return false;
    const A* selected = nullptr;
    for (const auto& a : d.audio)
    {
        if (audioId.isNotEmpty() && a.id == audioId && !samePath(path, a.relativePath))
            return fail(e, "Audio identity belongs to another path: " + audioId);
        if (!samePath(path, a.relativePath) || (audioId.isNotEmpty() && a.id != audioId)) continue;
        if (selected) return fail(e, "Ambiguous audio annotation variants; select audioId for: " + path);
        selected = &a;
    }
    if (!selected) return fail(e, "No audio section for path/audioId: " + path + " / " + audioId);
    rows = selected->rows;
    if (loadedFingerprint) *loadedFingerprint = fingerprintToken(d);
    remember(manifest, audio, d);
    return true;
}
bool OttoMaterialIO::listAudioVariants(const juce::File& manifest, const juce::File& audio,
                                      std::vector<Audio>& variants, S& e)
{
    e.clear(); S path; if (!relativeAudio(manifest, audio, path, e)) return false;
    D d; if (!loadDocument(manifest, d, e)) return false;
    std::vector<Audio> result;
    for (const auto& a : d.audio) if (samePath(path, a.relativePath)) result.push_back(a);
    variants = std::move(result);
    return true;
}
bool OttoMaterialIO::containsAudio(const juce::File& manifest, const juce::File& audio, S& e)
{
    e.clear(); S path; if (!relativeAudio(manifest, audio, path, e)) return false;
    D d; if (!loadDocument(manifest, d, e)) return false;
    remember(manifest, audio, d);
    for (const auto& a : d.audio) if (samePath(path, a.relativePath)) return true;
    return false; // A valid manifest does not suppress legacy data for other audio.
}
bool OttoMaterialIO::saveAudio(const juce::File& manifest, const juce::File& audio,
                                const std::vector<SampleRegionSetting>& rows, S& e, const S& expectedFingerprint,
                                const S& audioId)
{
    e.clear(); S path; if (!relativeAudio(manifest, audio, path, e)) return false;
    D d; if (manifest.exists() && !loadDocument(manifest, d, e)) return false;
    if (expectedFingerprint.isNotEmpty())
    {
        const auto current = fingerprintToken(d);
        if (expectedFingerprint != current) return fail(e, "Manifest conflict with supplied fingerprint; reload before saving");
    }
    {
        std::lock_guard<std::mutex> lock(observationsMutex);
        const auto it = observations.find(observationKey(manifest, audio));
        if (expectedFingerprint.isEmpty() && it != observations.end() && it->second.token != fingerprintToken(d))
            return fail(e, "Manifest conflict since loadAudio; reload before saving");
    }
    if (!replaceAudio(d, manifest, audio, rows, e, audioId) || !saveDocument(manifest, d, e)) return false;
    remember(manifest, audio, d); return true;
}
bool OttoMaterialIO::replaceAudio(Document& document, const juce::File& manifest,
                                   const juce::File& audio, const std::vector<SampleRegionSetting>& rows, S& e,
                                    const S& audioId)
{
    e.clear(); S path; if (!relativeAudio(manifest, audio, path, e)) return false;
    if (!validate(document, e)) return false;
    D d = document;
    A* target = nullptr;
    S suppliedAudio = audioId;
    for (const auto& r : rows) if (r.materialAudioId.isNotEmpty())
    {
        if (suppliedAudio.isNotEmpty() && suppliedAudio != r.materialAudioId) return fail(e, "Inconsistent supplied audio IDs");
        suppliedAudio = r.materialAudioId;
    }
    for (auto& a : d.audio)
    {
        if (suppliedAudio.isNotEmpty())
        {
            if (a.id != suppliedAudio) continue;
            if (!samePath(path, a.relativePath)) return fail(e, "Audio identity belongs to another path: " + suppliedAudio);
            target = &a;
        }
        else if (samePath(path, a.relativePath))
        {
            if (target) return fail(e, "Ambiguous audio annotation variants; select audioId for: " + path);
            target = &a;
        }
    }
    if (!target)
    {
        A a; a.relativePath = path; a.id = suppliedAudio.isNotEmpty() ? suppliedAudio : mint('a');
        d.audio.push_back(std::move(a)); target = &d.audio.back();
    }
    if (suppliedAudio.isNotEmpty() && suppliedAudio != target->id) return fail(e, "Supplied audio ID conflicts with existing section");
    auto updated = rows; std::set<S> claimed;
    std::map<S, int> incomingNames;
    for (const auto& r : updated) ++incomingNames[r.name];
    for (auto& r : updated)
    {
        r.materialAudioId = target->id;
        if (r.materialRegionId.isEmpty() && incomingNames[r.name] == 1)
        {
            const SampleRegionSetting* match = nullptr; int count = 0;
            for (const auto& old : target->rows) if (old.name == r.name) { match = &old; ++count; }
            if (count == 1) r.materialRegionId = match->materialRegionId;
        }
        if (r.materialRegionId.isEmpty()) r.materialRegionId = mint('p');
        if (!claimed.insert(r.materialRegionId).second) return fail(e, "Duplicate supplied pronunciation ID");
    }
    target->rows = std::move(updated);
    if (!validate(d, e)) return false;
    document = std::move(d); return true;
}
}
