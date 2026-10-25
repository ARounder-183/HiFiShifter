#include "MelodyneImporter.h"

#include <juce_core/juce_core.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <map>
#include <set>
#include <tuple>
#include <unordered_map>

namespace hachi::backend
{
namespace
{
constexpr std::size_t maxGraphBytes = 256u * 1024u * 1024u;
constexpr std::uint32_t nullReference = 0xffffffffu;

bool hasVoicedPitch(const NoteData& note)
{
    return std::any_of(note.contour.begin(), note.contour.end(),
                       [](const auto& point) { return point.voiced; });
}

template <typename T>
std::optional<T> readLittle(const std::vector<std::uint8_t>& data, std::size_t offset)
{
    if (offset > data.size() || sizeof(T) > data.size() - offset) return std::nullopt;
    T value {};
    std::memcpy(&value, data.data() + offset, sizeof(T));
    return value;
}

bool readExact(juce::InputStream& stream, void* destination, int bytes)
{
    return stream.read(destination, bytes) == bytes;
}

std::optional<std::vector<std::uint8_t>> decodeGraph(const juce::File& file, juce::String& error,
                                                     const MelodyneImporter::Progress& progress)
{
    juce::FileInputStream stream(file);
    if (!stream.openedOk()) { error = "Could not open MPD"; return std::nullopt; }
    std::uint8_t header[8] {};
    if (!readExact(stream, header, 8) || std::memcmp(header, "GNBCFA", 6) != 0)
    {
        error = "Not a Melodyne GNBCFA project";
        return std::nullopt;
    }
    const auto fileLength = std::max<juce::int64>(1, stream.getTotalLength());
    while (!stream.isExhausted() && stream.getPosition() < fileLength)
    {
        if (progress) progress(0.03 + 0.09 * static_cast<double>(stream.getPosition())
                                        / static_cast<double>(fileLength), "scan_container");
        std::uint8_t entryHeader[8] {};
        if (!readExact(stream, entryHeader, 8)) break;
        std::uint32_t nameLength {};
        std::memcpy(&nameLength, entryHeader, 4);
        if (nameLength > 1024u * 1024u) { error = "Invalid MPD entry name"; return std::nullopt; }
        std::vector<char> name(nameLength + 1u, 0);
        if (!readExact(stream, name.data(), static_cast<int>(nameLength)))
        { error = "Truncated MPD entry name"; return std::nullopt; }
        const auto aligned = (stream.getPosition() + 7) & ~juce::int64(7);
        if (!stream.setPosition(aligned)) { error = "Invalid MPD entry alignment"; return std::nullopt; }
        std::uint64_t storedLength {};
        if (!readExact(stream, &storedLength, 8) || storedLength > maxGraphBytes + 20u)
        { error = "Invalid MPD entry length"; return std::nullopt; }
        const auto payloadStart = stream.getPosition();
        const auto entryName = juce::String::fromUTF8(name.data()).toLowerCase();
        if (!entryName.contains("melodyne.graph"))
        {
            const auto nextEntry = (payloadStart + static_cast<juce::int64>(storedLength) + 7)
                & ~juce::int64(7);
            if (!stream.setPosition(nextEntry))
            { error = "Truncated MPD container"; return std::nullopt; }
            continue;
        }

        std::vector<std::uint8_t> stored(static_cast<std::size_t>(storedLength));
        if (!readExact(stream, stored.data(), static_cast<int>(stored.size())))
        { error = "Truncated MPD graph"; return std::nullopt; }
        if (stored.size() >= 20u && std::memcmp(stored.data(), "GNBKVAi\0", 8) == 0)
        {
            std::uint32_t compressedLength {};
            std::memcpy(&compressedLength, stored.data() + 16, 4);
            if (compressedLength != stored.size() - 20u)
            { error = "Invalid MPD compressed graph"; return std::nullopt; }
            if (progress) progress(0.13, "decompress_graph");
            juce::MemoryInputStream compressed(stored.data() + 20, compressedLength, false);
            juce::GZIPDecompressorInputStream inflater(&compressed, false,
                juce::GZIPDecompressorInputStream::zlibFormat);
            std::vector<std::uint8_t> decoded;
            std::array<std::uint8_t, 64 * 1024> buffer {};
            while (!inflater.isExhausted())
            {
                const auto count = inflater.read(buffer.data(), static_cast<int>(buffer.size()));
                if (count <= 0) break;
                if (decoded.size() + static_cast<std::size_t>(count) > maxGraphBytes)
                { error = "MPD graph exceeds decode limit"; return std::nullopt; }
                decoded.insert(decoded.end(), buffer.begin(), buffer.begin() + count);
            }
            if (decoded.empty()) { error = "MPD graph decompression failed"; return std::nullopt; }
            return decoded;
        }
        return stored;
    }
    error = "Melodyne project contains no object graph";
    return std::nullopt;
}

struct FieldSchema { std::string name; std::uint32_t type = 0; };
struct ClassSchema { std::string name; std::vector<FieldSchema> fields; };

struct RawValue
{
    enum class Kind { reference, boolean, integer, number } kind = Kind::integer;
    std::uint32_t reference = nullReference;
    std::int64_t integer = 0;
    double number = 0.0;
    bool boolean = false;
};

std::optional<std::size_t> valueSize(std::uint32_t type)
{
    switch (type)
    {
        case 0x85e: return 8;
        case 0x162: case 0x163: return 1;
        case 0x469: case 0x466: return 4;
        case 0x864: case 0x86c: case 0x871: return 8;
        case 0x1052: return 16;
        default: return std::nullopt;
    }
}

class Graph
{
public:
    bool parse(std::vector<std::uint8_t> bytes, juce::String& error)
    {
        data = std::move(bytes);
        std::size_t offset = 0;
        const auto takeU32 = [&]() -> std::optional<std::uint32_t>
        {
            const auto value = readLittle<std::uint32_t>(data, offset);
            if (value) offset += 4;
            return value;
        };
        const auto keyCount = takeU32();
        if (!keyCount || *keyCount == 0 || *keyCount > 16384) { error = "Invalid MPD keys"; return false; }
        std::vector<std::string> keys;
        keys.reserve(*keyCount);
        for (std::uint32_t i = 0; i < *keyCount; ++i)
        {
            const auto length = takeU32();
            if (!length || offset + *length > data.size()) { error = "Truncated MPD key"; return false; }
            keys.emplace_back(reinterpret_cast<const char*>(data.data() + offset), *length);
            offset += *length;
        }
        const auto classCount = takeU32();
        if (!classCount || *classCount == 0 || *classCount > 2048) { error = "Invalid MPD classes"; return false; }
        classes.reserve(*classCount);
        for (std::uint32_t i = 0; i < *classCount; ++i)
        {
            const auto nameLength = takeU32();
            if (!nameLength || offset + *nameLength > data.size()) { error = "Truncated MPD class"; return false; }
            ClassSchema schema;
            schema.name.assign(reinterpret_cast<const char*>(data.data() + offset), *nameLength);
            offset += *nameLength;
            if (!takeU32()) { error = "Truncated MPD class version"; return false; }
            const auto fieldCount = takeU32();
            if (!fieldCount || *fieldCount > 16384) { error = "Invalid MPD fields"; return false; }
            for (std::uint32_t field = 0; field < *fieldCount; ++field)
            {
                const auto key = takeU32();
                const auto multiplicity = takeU32();
                const auto type = takeU32();
                if (!key || !multiplicity || !type || *key >= keys.size() || offset >= data.size()
                    || !valueSize(*type))
                { error = "Unsupported MPD field schema"; return false; }
                ++offset;
                schema.fields.push_back({ keys[*key], *type });
            }
            classes.push_back(std::move(schema));
        }
        const auto objectCount = takeU32();
        if (!objectCount || *objectCount == 0 || *objectCount > 2000000)
        { error = "Invalid MPD object count"; return false; }
        objectClasses.reserve(*objectCount);
        for (std::uint32_t i = 0; i < *objectCount; ++i)
        {
            const auto classId = takeU32();
            if (!classId || (*classId >= classes.size() && *classId != nullReference))
            { error = "Invalid MPD object class"; return false; }
            objectClasses.push_back(*classId);
        }
        const auto serializedCount = takeU32();
        if (!serializedCount || *serializedCount > *objectCount) { error = "Invalid MPD records"; return false; }
        std::vector<std::uint32_t> ids;
        ids.reserve(*serializedCount);
        for (std::uint32_t i = 0; i < *serializedCount; ++i)
        {
            const auto id = takeU32();
            if (!id || *id >= *objectCount) { error = "Invalid MPD object id"; return false; }
            ids.push_back(*id);
        }
        records.resize(*objectCount);
        for (const auto id : ids)
        {
            const auto start = offset;
            const auto length = takeU32();
            if (!length || offset + *length > data.size()) { error = "Truncated MPD object"; return false; }
            records[id] = std::pair<std::size_t, std::size_t> { start, offset + *length };
            offset += *length;
        }
        return true;
    }

    std::size_t objectCount() const { return objectClasses.size(); }
    const ClassSchema* objectClass(std::uint32_t id) const
    {
        if (id >= objectClasses.size() || objectClasses[id] >= classes.size()) return nullptr;
        return &classes[objectClasses[id]];
    }
    std::string className(std::uint32_t id) const
    {
        if (const auto* schema = objectClass(id)) return schema->name;
        return {};
    }
    std::optional<RawValue> value(std::uint32_t id, const std::string& wanted) const
    {
        if (id >= records.size() || !records[id]) return std::nullopt;
        const auto* schema = objectClass(id);
        if (schema == nullptr) return std::nullopt;
        auto offset = records[id]->first + 8u;
        for (const auto& field : schema->fields)
        {
            const auto size = valueSize(field.type);
            if (!size || offset + *size > records[id]->second) return std::nullopt;
            if (field.name == wanted)
            {
                RawValue result;
                if (field.type == 0x85e)
                {
                    result.kind = RawValue::Kind::reference;
                    result.reference = readLittle<std::uint32_t>(data, offset).value_or(nullReference);
                }
                else if (field.type == 0x162 || field.type == 0x163)
                {
                    result.kind = RawValue::Kind::boolean;
                    result.boolean = data[offset] != 0;
                }
                else if (field.type == 0x469)
                {
                    result.kind = RawValue::Kind::integer;
                    result.integer = readLittle<std::int32_t>(data, offset).value_or(0);
                }
                else if (field.type == 0x86c)
                {
                    result.kind = RawValue::Kind::integer;
                    result.integer = readLittle<std::int64_t>(data, offset).value_or(0);
                }
                else if (field.type == 0x466)
                {
                    result.kind = RawValue::Kind::number;
                    result.number = readLittle<float>(data, offset).value_or(0.0f);
                }
                else if (field.type == 0x864)
                {
                    result.kind = RawValue::Kind::number;
                    result.number = readLittle<double>(data, offset).value_or(0.0);
                }
                else if (field.type == 0x871)
                {
                    const auto numerator = readLittle<std::int32_t>(data, offset).value_or(0);
                    const auto denominator = readLittle<std::int32_t>(data, offset + 4).value_or(0);
                    result.kind = RawValue::Kind::number;
                    result.number = denominator == 0 ? 0.0 : static_cast<double>(numerator) / denominator;
                }
                else return std::nullopt;
                return result;
            }
            offset += *size;
        }
        return std::nullopt;
    }
    std::optional<std::uint32_t> reference(std::uint32_t id, const std::string& field) const
    {
        const auto item = value(id, field);
        if (!item || item->kind != RawValue::Kind::reference || item->reference == nullReference) return std::nullopt;
        return item->reference;
    }
    std::optional<double> number(std::uint32_t id, const std::string& field) const
    {
        const auto item = value(id, field);
        if (!item) return std::nullopt;
        if (item->kind == RawValue::Kind::number) return std::isfinite(item->number) ? std::optional(item->number) : std::nullopt;
        if (item->kind == RawValue::Kind::integer) return static_cast<double>(item->integer);
        return std::nullopt;
    }
    std::optional<double> numberAlias(std::uint32_t id,
                                      std::initializer_list<const char*> fields) const
    {
        for (const auto* field : fields)
            if (const auto result = number(id, field)) return result;
        return std::nullopt;
    }
    bool boolean(std::uint32_t id, const std::string& field) const
    {
        const auto item = value(id, field);
        return item && item->kind == RawValue::Kind::boolean && item->boolean;
    }
    bool booleanAlias(std::uint32_t id,
                      std::initializer_list<const char*> fields) const
    {
        for (const auto* field : fields)
            if (boolean(id, field)) return true;
        return false;
    }
    std::vector<std::uint32_t> list(std::uint32_t id) const
    {
        std::vector<std::uint32_t> output;
        if (className(id) != "GNList" || id >= records.size() || !records[id]) return output;
        const auto start = records[id]->first;
        const auto count = readLittle<std::uint32_t>(data, start + 29u).value_or(0);
        if (count > 2000000 || start + 33u + static_cast<std::size_t>(count) * 4u > records[id]->second)
            return output;
        output.reserve(count);
        for (std::uint32_t index = 0; index < count; ++index)
            output.push_back(readLittle<std::uint32_t>(data, start + 33u + index * 4u).value_or(nullReference));
        return output;
    }
    juce::String string(std::uint32_t id) const
    {
        if (className(id) != "GNString" || id >= records.size() || !records[id]) return {};
        const auto start = records[id]->first;
        const auto fixed = readLittle<std::uint32_t>(data, start + 4u).value_or(0);
        const auto extra = start + 8u + fixed;
        const auto length = readLittle<std::uint32_t>(data, extra + 12u).value_or(0);
        const auto rawStart = extra + 20u;
        if (rawStart + length > records[id]->second) return {};
        const auto encoding = static_cast<int>(number(id, "encoding").value_or(1.0));
        if (encoding == 5)
        {
            std::vector<juce::CharPointer_UTF16::CharType> units;
            for (std::size_t index = 0; index + 1u < length; index += 2u)
            {
                const auto unit = readLittle<std::uint16_t>(data, rawStart + index).value_or(0);
                if (unit == 0) break;
                units.push_back(static_cast<juce::CharPointer_UTF16::CharType>(unit));
            }
            units.push_back(0);
            return juce::String(juce::CharPointer_UTF16(units.data())).trim();
        }
        return juce::String::fromUTF8(reinterpret_cast<const char*>(data.data() + rawStart),
                                      static_cast<int>(length)).trim();
    }
    juce::String stringField(std::uint32_t id, const std::string& field) const
    {
        const auto ref = reference(id, field);
        return ref ? string(*ref) : juce::String{};
    }
    std::optional<std::uint32_t> firstClass(const std::string& name) const
    {
        for (std::uint32_t id = 0; id < objectClasses.size(); ++id)
            if (className(id) == name) return id;
        return std::nullopt;
    }
    std::vector<std::pair<double, double>> functionPoints(std::uint32_t function) const
    {
        if (className(function) == "MUConstantFunction")
            return { { 0.0, number(function, "y").value_or(0.0) } };
        const auto pointsRef = reference(function, "points");
        if (!pointsRef) return {};
        std::vector<std::pair<double, double>> points;
        for (const auto point : list(*pointsRef))
            if (const auto x = number(point, "x"))
                if (const auto y = number(point, "y")) points.emplace_back(*x, *y);
        std::sort(points.begin(), points.end());
        return points;
    }
    double evaluate(std::uint32_t function, double x) const
    {
        if (className(function) == "MUBezierDataPointFunction")
        {
            const auto pointsRef = reference(function, "points");
            const auto points = pointsRef ? list(*pointsRef) : std::vector<std::uint32_t>{};
            if (points.empty()) return x;
            const auto firstX = number(points.front(), "x").value_or(0.0);
            if (x <= firstX || points.size() == 1)
                return number(points.front(), "y").value_or(x);
            for (std::size_t index = 1; index < points.size(); ++index)
            {
                const auto left = points[index - 1];
                const auto right = points[index];
                const auto x0 = number(left, "x").value_or(0.0);
                const auto y0 = number(left, "y").value_or(x0);
                const auto x3 = number(right, "x").value_or(x0);
                const auto y3 = number(right, "y").value_or(y0);
                if (x > x3) continue;
                const auto x1 = x0 + number(left, "rightControlPointXDelta").value_or(0.0);
                const auto y1 = y0 + number(left, "rightControlPointYDelta").value_or(0.0);
                const auto x2 = x3 + number(right, "leftControlPointXDelta").value_or(0.0);
                const auto y2 = y3 + number(right, "leftControlPointYDelta").value_or(0.0);
                auto low = 0.0;
                auto high = 1.0;
                for (int iteration = 0; iteration < 28; ++iteration)
                {
                    const auto t = (low + high) * 0.5;
                    const auto u = 1.0 - t;
                    const auto bezierX = u * u * u * x0 + 3.0 * u * u * t * x1
                        + 3.0 * u * t * t * x2 + t * t * t * x3;
                    if (bezierX < x) low = t; else high = t;
                }
                const auto t = (low + high) * 0.5;
                const auto u = 1.0 - t;
                return u * u * u * y0 + 3.0 * u * u * t * y1
                    + 3.0 * u * t * t * y2 + t * t * t * y3;
            }
            return number(points.back(), "y").value_or(x);
        }
        const auto points = functionPoints(function);
        if (points.empty()) return x;
        if (points.size() == 1 || x <= points.front().first) return points.front().second;
        for (std::size_t index = 1; index < points.size(); ++index)
            if (x <= points[index].first)
            {
                const auto span = std::max(1.0e-9, points[index].first - points[index - 1].first);
                const auto amount = std::clamp((x - points[index - 1].first) / span, 0.0, 1.0);
                return points[index - 1].second + (points[index].second - points[index - 1].second) * amount;
            }
        return points.back().second;
    }
    std::vector<std::pair<double, double>> sampledFunctionPoints(
        std::uint32_t function) const
    {
        const auto anchors = functionPoints(function);
        if (anchors.size() < 2) return anchors;
        // Melodyne's Bezier time warp can change slope sharply around Attack.
        // A bounded 2 ms grid retains those changes without allowing one long
        // element to allocate an unbounded number of project points.
        const auto span = std::abs(anchors.back().first - anchors.front().first);
        const auto steps = std::clamp(static_cast<std::size_t>(std::ceil(span / 0.002)),
                                      std::size_t(2), std::size_t(2048));
        std::vector<std::pair<double, double>> sampled;
        sampled.reserve(steps + 1);
        for (std::size_t index = 0; index <= steps; ++index)
        {
            const auto amount = static_cast<double>(index) / static_cast<double>(steps);
            const auto target = anchors.front().first
                + (anchors.back().first - anchors.front().first) * amount;
            sampled.emplace_back(target, evaluate(function, target));
        }
        return sampled;
    }
    double inverse(std::uint32_t function, double y) const
    {
        const auto points = functionPoints(function);
        if (points.empty()) return y;
        auto low = points.front().first;
        auto high = points.back().first;
        const auto ascending = points.back().second >= points.front().second;
        for (int iteration = 0; iteration < 32; ++iteration)
        {
            const auto middle = (low + high) * 0.5;
            if ((evaluate(function, middle) < y) == ascending) low = middle; else high = middle;
        }
        return (low + high) * 0.5;
    }

private:
    std::vector<std::uint8_t> data;
    std::vector<ClassSchema> classes;
    std::vector<std::uint32_t> objectClasses;
    std::vector<std::optional<std::pair<std::size_t, std::size_t>>> records;
};

juce::String makeId(const char* prefix)
{
    return juce::String(prefix) + "_" + juce::Uuid().toString().removeCharacters("-");
}

std::optional<std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>>
sourceChain(const Graph& graph, std::uint32_t element)
{
    const auto componentsRef = graph.reference(element, "audioComponents");
    const auto components = componentsRef ? graph.list(*componentsRef) : std::vector<std::uint32_t>{};
    auto component = graph.reference(element, "principalAudioComponent");
    if (!component && !components.empty()) component = components.front();
    if (!component) return std::nullopt;
    const auto sourceComponent = graph.reference(*component, "audioSourceComponent");
    if (!sourceComponent) return std::nullopt;
    const auto item = graph.reference(*sourceComponent, "audioSourceItem");
    const auto sourceElement = graph.reference(*sourceComponent, "audioSourceElement");
    if (!item || !sourceElement) return std::nullopt;
    const auto description = graph.reference(*sourceElement, "audioSourceDescription");
    if (!description) return std::nullopt;
    const auto source = graph.reference(*description, "audioSource");
    if (!source) return std::nullopt;
    return std::tuple { *source, *item, *description };
}

juce::File resolveMedia(const juce::String& stored, const juce::File& projectDirectory,
                        const std::map<juce::String, juce::File>& media)
{
    const auto normalized = stored.replaceCharacter('\\', '/');
#if JUCE_LINUX
    // MPD stores Windows absolute paths even when opened through WSL. Resolve
    // the exact drive path before basename search (which can select a different
    // recording with the same name).
    if (normalized.length() >= 3 && normalized[1] == ':' && normalized[2] == '/')
    {
        const auto drive = normalized.substring(0, 1).toLowerCase();
        const auto mounted = juce::File("/mnt").getChildFile(drive)
            .getChildFile(normalized.substring(3));
        if (mounted.existsAsFile()) return mounted;
    }
#endif
    if (juce::File::isAbsolutePath(normalized))
    {
        const juce::File direct(normalized);
        if (direct.existsAsFile()) return direct;
    }
    const auto relative = projectDirectory.getChildFile(normalized);
    if (relative.existsAsFile()) return relative;
    const auto name = juce::File(normalized).getFileName();
    const auto beside = projectDirectory.getChildFile(name);
    if (beside.existsAsFile()) return beside;
    if (const auto found = media.find(name.toLowerCase()); found != media.end()) return found->second;
    return {};
}

std::map<juce::String, juce::File> buildMediaIndex(const juce::File& directory)
{
    juce::Array<juce::File> files;
    directory.findChildFiles(files, juce::File::findFiles, true);
    std::sort(files.begin(), files.end(), [](const auto& left, const auto& right)
    { return left.getFullPathName() < right.getFullPathName(); });
    std::map<juce::String, juce::File> result;
    for (const auto& file : files) result.try_emplace(file.getFileName().toLowerCase(), file);
    return result;
}

void collectTracks(const Graph& graph, std::uint32_t track, std::vector<std::uint32_t>& result,
                   std::set<std::uint32_t>& seen)
{
    if (!seen.insert(track).second) return;
    if (const auto elements = graph.reference(track, "elements"); elements && !graph.list(*elements).empty())
        result.push_back(track);
    if (const auto children = graph.reference(track, "subtracks"))
        for (const auto child : graph.list(*children)) collectTracks(graph, child, result, seen);
}

double projectBpm(const Graph& graph)
{
    for (std::uint32_t id = 0; id < graph.objectCount(); ++id)
        if (const auto* schema = graph.objectClass(id))
            for (const auto& field : schema->fields)
            {
                const auto lower = juce::String(field.name).toLowerCase();
                const auto value = graph.number(id, field.name);
                if (!value || *value <= 0.0) continue;
                if ((lower == "beatsperminute" || lower == "bpm" || lower.contains("tempo"))
                    && *value >= 10.0 && *value <= 400.0) return *value;
                if ((lower.contains("secondsperbeat") || lower.contains("quarterduration"))
                    && *value >= 0.15 && *value <= 6.0) return 60.0 / *value;
            }
    if (const auto timeline = graph.firstClass("MUQuarterTimeline"))
        if (const auto anchors = graph.reference(*timeline, "quarterAnchors"))
        {
            const auto points = graph.list(*anchors);
            for (std::size_t i = 1; i < points.size(); ++i)
            {
                const auto t0 = graph.number(points[i - 1], "timePosition");
                const auto t1 = graph.number(points[i], "timePosition");
                const auto q0 = graph.number(points[i - 1], "quarterPosition");
                const auto q1 = graph.number(points[i], "quarterPosition");
                if (t0 && t1 && q0 && q1 && *t1 > *t0 && *q1 > *q0)
                {
                    const auto bpm = 60.0 * (*q1 - *q0) / (*t1 - *t0);
                    if (bpm >= 10.0 && bpm <= 400.0) return bpm;
                }
            }
        }
    return 120.0;
}

double projectBeatOrigin(const Graph& graph, double bpm)
{
    if (const auto timeline = graph.firstClass("MUQuarterTimeline"))
        if (const auto anchors = graph.reference(*timeline, "quarterAnchors"))
            for (const auto anchor : graph.list(*anchors))
            {
                const auto time = graph.number(anchor, "timePosition");
                const auto quarter = graph.number(anchor, "quarterPosition");
                if (time && quarter) return *time - *quarter * 60.0 / std::max(1.0, bpm);
            }
    return 0.0;
}

struct SourcePitchPoint { double slice = 0.0; float raw = 0.0f; float smooth = 0.0f; bool silent = false; };

std::optional<std::pair<float, float>> pitchAt(const std::vector<SourcePitchPoint>& points, double slice)
{
    if (points.empty() || slice < points.front().slice - 0.500001 || slice > points.back().slice + 0.500001)
        return std::nullopt;
    const auto right = std::lower_bound(points.begin(), points.end(), slice,
        [](const auto& point, double value) { return point.slice < value; });
    const auto rightIndex = static_cast<std::size_t>(right == points.end() ? points.size() - 1 : right - points.begin());
    const auto leftIndex = rightIndex > 0 && points[rightIndex].slice > slice ? rightIndex - 1 : rightIndex;
    const auto& left = points[leftIndex];
    const auto& next = points[rightIndex];
    if (left.silent || next.silent) return std::nullopt;
    const auto amount = next.slice > left.slice
        ? static_cast<float>(std::clamp((slice - left.slice) / (next.slice - left.slice), 0.0, 1.0)) : 0.0f;
    const auto interpolate = [&](auto value)
    {
        const auto linear = value(left) + (value(next) - value(left)) * amount;
        if (rightIndex == leftIndex) return linear;
        const auto& before = points[leftIndex > 0 ? leftIndex - 1 : leftIndex];
        const auto& after = points[std::min(rightIndex + 1, points.size() - 1)];
        if (before.silent || after.silent) return linear;
        const auto y0 = value(before);
        const auto y1 = value(left);
        const auto y2 = value(next);
        const auto y3 = value(after);
        const auto x2 = amount * amount;
        const auto x3 = x2 * amount;
        const auto tangent1 = 0.5f * (y2 - y0);
        const auto tangent2 = 0.5f * (y3 - y1);
        const auto cubic = (2.0f * x3 - 3.0f * x2 + 1.0f) * y1
            + (x3 - 2.0f * x2 + amount) * tangent1
            + (-2.0f * x3 + 3.0f * x2) * y2
            + (x3 - x2) * tangent2;
        // Catmull-Rom tangents provide a smooth derivative, while this bound
        // prevents overshoot from inventing a pitch excursion that is absent
        // from every persisted Melodyne analysis point.
        return std::clamp(cubic, std::min({ y0, y1, y2, y3 }),
                          std::max({ y0, y1, y2, y3 }));
    };
    return std::pair { interpolate([](const auto& point) { return point.raw; }),
                       interpolate([](const auto& point) { return point.smooth; }) };
}
}

std::vector<MelodyneConsonantMapping>
MelodyneImporter::consonantCandidates(const std::vector<NoteData>& notes)
{
    std::vector<MelodyneConsonantMapping> mappings;
    for (std::size_t index = 0; index + 1 < notes.size(); ++index)
    {
        const auto& consonant = notes[index];
        const auto& vowel = notes[index + 1];
        const auto adjacent = std::abs(consonant.startSeconds + consonant.durationSeconds
                                        - vowel.startSeconds) <= 0.002;
        if (!adjacent || hasVoicedPitch(consonant) || !hasVoicedPitch(vowel)) continue;
        mappings.push_back({ consonant.id, vowel.id, consonant.durationSeconds });
    }
    return mappings;
}

juce::var MelodyneImporter::inspectTracks(const juce::File& file, juce::String& error)
{
    auto bytes = decodeGraph(file, error, {});
    Graph graph;
    if (!bytes || !graph.parse(std::move(*bytes), error)) return {};
    juce::Array<juce::var> tracks;
    for (std::uint32_t id = 0; id < graph.objectCount(); ++id)
    {
        const auto className = juce::String(graph.className(id));
        if (!className.containsIgnoreCase("track")) continue;
        auto* row = new juce::DynamicObject();
        row->setProperty("object_id", static_cast<int>(id));
        row->setProperty("class", className);
        row->setProperty("title", graph.stringField(id, "title"));
        row->setProperty("name", graph.stringField(id, "name"));
        const auto elements = graph.reference(id, "elements");
        row->setProperty("elements", elements ? static_cast<int>(graph.list(*elements).size()) : 0);
        juce::StringArray media;
        if (elements)
            for (const auto element : graph.list(*elements))
                if (const auto chain = sourceChain(graph, element))
                {
                    const auto [source, item, description] = *chain;
                    juce::ignoreUnused(item, description);
                    if (const auto path = graph.reference(source, "filePath"))
                        media.addIfNotAlreadyThere(graph.stringField(*path, "posixPath"));
                }
        row->setProperty("media", media.joinIntoString("\n"));
        const auto children = graph.reference(id, "subtracks");
        row->setProperty("subtracks", children ? static_cast<int>(graph.list(*children).size()) : 0);
        tracks.add(juce::var(row));
    }
    return tracks;
}

std::optional<MelodyneImportResult> MelodyneImporter::importProject(
    const juce::File& file, juce::String& error, Progress progress,
    MelodyneImportOptions options)
{
    if (progress) progress(0.01, "open");
    auto bytes = decodeGraph(file, error, progress);
    if (!bytes) return std::nullopt;
    Graph graph;
    if (!graph.parse(std::move(*bytes), error)) return std::nullopt;
    if (progress) progress(0.25, "read_tracks");

    const auto performance = graph.firstClass("MUPerformance");
    if (!performance) { error = "Melodyne graph has no performance"; return std::nullopt; }
    const auto rootTrack = graph.reference(*performance, "rootTrack");
    if (!rootTrack) { error = "Melodyne performance has no root track"; return std::nullopt; }

    std::map<std::uint32_t, juce::String> sourcePaths;
    if (const auto sourceList = graph.reference(*performance, "audioSources"))
        for (const auto source : graph.list(*sourceList))
            if (graph.className(source) == "MUAudioFileSource")
                if (const auto path = graph.reference(source, "filePath"))
                {
                    const auto value = graph.stringField(*path, "posixPath");
                    if (value.isNotEmpty()) sourcePaths[source] = value;
                }

    std::vector<std::uint32_t> trackIds;
    std::set<std::uint32_t> seen;
    collectTracks(graph, *rootTrack, trackIds, seen);
    if (trackIds.empty()) { error = "Melodyne project contains no playable tracks"; return std::nullopt; }

    const auto mediaIndex = options.recursiveMediaSearch
        ? buildMediaIndex(file.getParentDirectory())
        : std::map<juce::String, juce::File>{};
    std::map<std::uint32_t, juce::File> resolved;
    MelodyneImportResult result;
    result.project.name = file.getFileNameWithoutExtension();
    result.project.bpm = projectBpm(graph);
    for (const auto& [source, path] : sourcePaths)
    {
        result.referencedFiles.add(path);
        const auto media = resolveMedia(path, file.getParentDirectory(), mediaIndex);
        if (media.existsAsFile()) resolved[source] = media;
        else result.missingFiles.add(path);
    }

    double earliest = 0.0;
    for (const auto trackId : trackIds)
        if (const auto list = graph.reference(trackId, "elements"))
            for (const auto element : graph.list(*list))
                earliest = std::min(earliest, graph.number(element, "startTime").value_or(0.0));
    const auto shift = earliest < 0.0 ? -earliest : 0.0;
    result.project.beatOriginSeconds = projectBeatOrigin(graph, result.project.bpm) + shift;

    for (std::size_t trackIndex = 0; trackIndex < trackIds.size(); ++trackIndex)
    {
        if (progress) progress(0.35 + 0.55 * static_cast<double>(trackIndex) / trackIds.size(), "create_tracks");
        const auto trackId = trackIds[trackIndex];
        TrackData track;
        track.id = makeId("track");
        track.name = graph.stringField(trackId, "title");
        if (track.name.isEmpty()) track.name = "Melodyne Track " + juce::String(trackIndex + 1);
        track.muted = graph.boolean(trackId, "isMuted");
        track.solo = graph.boolean(trackId, "isSolo");
        track.volume = static_cast<float>(std::clamp(graph.number(trackId, "volume").value_or(1.0), 0.0, 4.0));
        if (!options.preserveProjectEdits)
        {
            track.muted = false;
            track.solo = false;
            track.volume = 1.0f;
        }
        const auto analyzer = graph.stringField(trackId, "defaultAnalyzerParameterSetIdenfier").toLowerCase();
        track.compose = analyzer.contains(".melodic");
        track.pitchAlgorithm = defaultPitchAlgorithm();
        track.stretchAlgorithm = StretchAlgorithm::melodyneHybrid;

        const auto elementList = graph.reference(trackId, "elements");
        if (!elementList) continue;
        const auto elementIds = graph.list(*elementList);
        // Keep the real imported endpoint identity until all playable elements
        // exist. This is annotation provenance, independent of render joins.
        std::map<std::uint32_t, std::pair<std::uint32_t, std::size_t>> importedSourceUses;
        std::set<std::uint32_t> connectedPrevious;
        for (const auto element : elementIds)
            if (const auto join = graph.reference(element, "followingJoin"))
                if (graph.boolean(*join, "joinsPitches"))
                    if (const auto following = graph.reference(*join, "followingElement"))
                        connectedPrevious.insert(*following);
        for (const auto element : elementIds)
        {
            if (graph.className(element) != "MUElement") continue;
            const auto chain = sourceChain(graph, element);
            if (!chain) continue;
            const auto [source, item, description] = *chain;
            if (const auto parameterSet = graph.reference(description, "analyzerParameterSet"))
            {
                const auto identifier = graph.stringField(*parameterSet, "identifier").toLowerCase();
                if (identifier.contains(".melodic") || graph.boolean(*parameterSet, "isTonalicOnly"))
                    track.compose = true;
            }
            const auto media = resolved.find(source);
            if (media == resolved.end()) continue;
            const auto start = graph.number(element, "startTime").value_or(0.0) + shift;
            const auto duration = graph.number(element, "duration").value_or(0.0);
            if (!std::isfinite(start) || !std::isfinite(duration) || duration <= 0.0) continue;
            const auto sampleRate = std::max(1.0, graph.number(source, "sampleRate").value_or(44100.0));
            const auto itemStart = std::max(0.0, graph.number(item, "startSampleIndex").value_or(0.0)) / sampleRate;
            const auto timeFunction = graph.reference(element, "sourceTimeForElementTimeFunction");
            const auto mappedStart = timeFunction ? graph.evaluate(*timeFunction, 0.0) : 0.0;
            const auto mappedEnd = timeFunction ? graph.evaluate(*timeFunction, duration) : duration;
            const auto sourceStart = itemStart + std::max(0.0, mappedStart);
            const auto sourceEnd = itemStart + std::max(mappedStart + 0.001, mappedEnd);

            ClipData clip;
            clip.id = makeId("clip");
            clip.sourceFile = media->second;
            clip.startSeconds = start;
            clip.durationSeconds = duration;
            clip.sourceOffsetSeconds = sourceStart;
            clip.sourceDurationSeconds = sourceEnd - sourceStart;
            clip.muted = graph.boolean(element, "isMuted");
            clip.fadeInSeconds = std::clamp(graph.number(element, "fadeInTime").value_or(0.0), 0.0, duration);
            clip.fadeOutSeconds = std::clamp(graph.number(element, "fadeOutTime").value_or(0.0), 0.0, duration);
            if (clip.fadeInSeconds <= 1.0e-9)
                if (const auto sourceHandle = graph.number(element, "amplitudeFadeInEndSourceTime"))
                {
                    const auto localSource = std::max(0.0, *sourceHandle - itemStart);
                    clip.fadeInSeconds = std::clamp(timeFunction ? graph.inverse(*timeFunction, localSource)
                                                                  : localSource, 0.0, duration);
                }
            if (clip.fadeOutSeconds <= 1.0e-9)
                if (const auto sourceHandle = graph.number(element, "amplitudeFadeOutStartSourceTime"))
                {
                    const auto localSource = std::max(0.0, *sourceHandle - itemStart);
                    const auto startFade = std::clamp(timeFunction ? graph.inverse(*timeFunction, localSource)
                                                                   : localSource, 0.0, duration);
                    clip.fadeOutSeconds = duration - startFade;
                }
            clip.gain = static_cast<float>(std::max(0.0, graph.number(element, "amplitudeFactor").value_or(1.0)));

            // sourceTimeForElementTimeFunction is Melodyne's authoritative
            // element-time -> source-time warp.  Keeping only mappedStart and
            // mappedEnd turns every Attack/vowel adjustment into a linear
            // stretch, which moves F0 away from the waveform near note tails.
            // Preserve the original piecewise-linear control points plus exact
            // endpoints, normalised to this clip's selected source range.
            if (timeFunction)
            {
                std::vector<double> targetAnchors { 0.0, duration };
                for (const auto& [target, sourceTime] : graph.sampledFunctionPoints(*timeFunction))
                {
                    juce::ignoreUnused(sourceTime);
                    if (std::isfinite(target) && target > 0.0 && target < duration)
                        targetAnchors.push_back(target);
                }
                if (const auto attack = graph.number(element, "attackDuration");
                    attack && *attack > 0.0 && *attack < duration)
                    targetAnchors.push_back(*attack);
                std::sort(targetAnchors.begin(), targetAnchors.end());
                targetAnchors.erase(std::unique(targetAnchors.begin(), targetAnchors.end(),
                    [](double left, double right) { return std::abs(left - right) <= 1.0e-9; }),
                    targetAnchors.end());
                auto previousSource = -1.0;
                for (const auto target : targetAnchors)
                {
                    const auto relativeSource = std::clamp(
                        graph.evaluate(*timeFunction, target) - mappedStart,
                        0.0, clip.sourceDurationSeconds);
                    if (relativeSource + 1.0e-8 < previousSource) continue;
                    if (!clip.sourceTimeMap.empty()
                        && relativeSource <= previousSource + 1.0e-8
                        && target < duration - 1.0e-9)
                        continue;
                    clip.sourceTimeMap.push_back({ target, relativeSource });
                    previousSource = relativeSource;
                }
                if (clip.sourceTimeMap.size() < 2)
                    clip.sourceTimeMap.clear();
            }

            NoteData note;
            note.id = makeId("note");
            note.label = graph.stringField(element, "label");
            if (note.label.isEmpty()) note.label = graph.stringField(element, "noteName");
            if (note.label.isEmpty()) note.label = graph.stringField(item, "label");
            note.startSeconds = 0.0;
            note.durationSeconds = duration;
            // On Melodyne import the consonant line defaults to the note's
            // preutterance (which is zero for a fresh import), so it sits on the
            // note start rather than at Melodyne's attackDuration; the start and
            // cutoff region lines default to zero (no trim).  The user tunes
            // these by hand afterwards.  UTAU import/timing is untouched.
            note.consonantSeconds = std::max(0.0, note.utauPreutteranceSeconds);
            const auto targetCenter = static_cast<float>(graph.numberAlias(
                element, { "pitchCenter", "targetPitchCenter" }).value_or(6000.0));
            auto sourceCenter = static_cast<float>(graph.number(item, "pitchCenter").value_or(targetCenter));
            note.midiNote = std::clamp(targetCenter / 100.0f, 0.0f, 127.0f);
            note.sourceMidiCenter = std::clamp(sourceCenter / 100.0f, 0.0f, 127.0f);
            note.drift = std::clamp(static_cast<float>(graph.numberAlias(element,
                { "pitchDriftFactor", "pitchDrift", "driftFactor" }).value_or(1.0)),
                0.0f, 2.0f);
            note.modulation = std::clamp(static_cast<float>(graph.numberAlias(element,
                { "pitchModulationFactor", "pitchModulationAmplitudeFactor",
                  "pitchModulation", "modulationFactor" }).value_or(1.0)),
                0.0f, 2.0f);
            // These are element edits, not analysis defaults.  Dropping them
            // made a saved Melodyne voice colour disappear on import even
            // though pitch and timing were restored correctly.
            note.formantSemitones = static_cast<float>(std::clamp(
                graph.number(element, "formantOffset").value_or(0.0) / 100.0, -12.0, 12.0));
            note.breath = static_cast<float>(std::clamp(
                graph.number(element, "sibilantBalance").value_or(0.0), 0.0, 1.0));
            note.attackSpeed = static_cast<float>(std::max(1.0e-6,
                graph.number(element, "sourceTimeForElementTimeFunctionAttackSlope").value_or(1.0)));
            // Melodyne stores its element gain and fade limits separately from
            // the pitch object.  Carry the simple envelope into the native
            // layer so the HJM conversion does not lose note-level decay.
            const auto gainDb = note.gain > 1.0e-6f
                ? juce::jlimit(-60.0f, 12.0f,
                    static_cast<float>(20.0 * std::log10(note.gain))) : -60.0f;
            const auto fadeIn = juce::jlimit(0.0, duration, clip.fadeInSeconds);
            const auto fadeOut = juce::jlimit(0.0, duration, clip.fadeOutSeconds);
            note.amplitudeEnvelope = { { 0.0, fadeIn > 1.0e-9 ? -60.0f : gainDb },
                { fadeIn, gainDb },
                { std::max(fadeIn, duration - fadeOut), gainDb },
                { duration, fadeOut > 1.0e-9 ? -60.0f : gainDb } };
            const auto robustFields = { "robustPitchCurve", "robustPitchCurveSwitch",
                "robustPitchCurveEnabled", "useRobustPitchCurve",
                "usesRobustPitchCurve", "isRobustPitchCurveEnabled" };
            note.robustPitchCurve = graph.booleanAlias(element, robustFields)
                || graph.booleanAlias(item, robustFields)
                || graph.booleanAlias(source, robustFields)
                || graph.booleanAlias(description, robustFields);
            if (const auto parameterSet = graph.reference(description, "analyzerParameterSet"))
                note.robustPitchCurve = note.robustPitchCurve
                    || graph.booleanAlias(*parameterSet, robustFields);
            note.connectedToPrevious = connectedPrevious.contains(element);

            std::vector<SourcePitchPoint> propertyPoints;
            if (const auto propertyList = graph.reference(item, "propertyPoints"))
            {
                const auto pointIds = graph.list(*propertyList);
                propertyPoints.reserve(pointIds.size());
                for (std::size_t pointIndex = 0; pointIndex < pointIds.size(); ++pointIndex)
                {
                    const auto point = pointIds[pointIndex];
                    const auto raw = graph.number(point, "pitchCent");
                    if (!raw) continue;
                    propertyPoints.push_back({ graph.number(point, "timeSliceIndex").value_or(static_cast<double>(pointIndex)),
                        static_cast<float>(*raw), static_cast<float>(graph.number(point, "pitchWithoutVibrato").value_or(*raw)),
                        graph.boolean(point, "isConsideredSilent") });
                }
                std::sort(propertyPoints.begin(), propertyPoints.end(),
                    [](const auto& left, const auto& right) { return left.slice < right.slice; });
            }
            if (!propertyPoints.empty())
            {
                std::vector<float> centres;
                for (double local = 0.0; local <= duration; local += 0.01)
                {
                    const auto mapped = timeFunction ? graph.evaluate(*timeFunction, local) : local;
                    if (const auto pitch = pitchAt(propertyPoints, (itemStart + mapped) * sampleRate / 1024.0))
                        if (pitch->second > 100.0f) centres.push_back(pitch->second);
                }
                if (!centres.empty())
                {
                    std::sort(centres.begin(), centres.end());
                    const auto derived = centres[centres.size() / 2];
                    if (sourceCenter <= 100.0f || std::abs(sourceCenter - derived) > 400.0f) sourceCenter = derived;
                    note.sourceMidiCenter = std::clamp(sourceCenter / 100.0f, 0.0f, 127.0f);
                }
                for (double local = 0.0; local <= duration + 0.0025; local += 0.005)
                {
                    const auto mapped = timeFunction ? graph.evaluate(*timeFunction, std::min(local, duration))
                                                     : std::min(local, duration);
                    const auto pitch = pitchAt(propertyPoints, (itemStart + mapped) * sampleRate / 1024.0);
                    note.contour.push_back({ std::min(local, duration), pitch ? pitch->first - sourceCenter : 0.0f,
                                             pitch ? pitch->second - sourceCenter : 0.0f, pitch.has_value() });
                }
            }

            const auto itemSamples = graph.number(item, "sampleCount").value_or(0.0);
            const auto addSibilant = [&](const char* field)
            {
                const auto offset = graph.number(item, field);
                if (!offset || *offset <= 0.0 || *offset >= itemSamples) return;
                const auto sourceLocal = *offset / sampleRate;
                const auto local = timeFunction ? graph.inverse(*timeFunction, sourceLocal) : sourceLocal;
                if (local >= 0.0 && local <= duration) note.sibilantMarkers.push_back(local);
            };
            addSibilant("startSibilantEndSampleOffset");
            addSibilant("endSibilantStartSampleOffset");

            if (const auto join = graph.reference(element, "followingJoin"))
            {
                note.connectedToNext = graph.boolean(*join, "joinsPitches");
                // Melodyne's real crossfade: a LINEAR amplitude transition of
                // MUSuccessiveJoin.amplitudeTransitionDuration when amplitudes
                // are joined.  The element is placed exactly back-to-back with
                // its join partner, so each side ramps for half the duration.
                if (graph.boolean(*join, "joinsAmplitudes"))
                    if (const auto atd = graph.number(*join, "amplitudeTransitionDuration"))
                        clip.crossfadeOutSeconds = std::clamp(*atd * 0.5, 0.0, duration);
            }
            if (const auto join = graph.reference(element, "precedingJoin"))
            {
                if (graph.boolean(*join, "joinsAmplitudes"))
                    if (const auto atd = graph.number(*join, "amplitudeTransitionDuration"))
                        clip.crossfadeInSeconds = std::clamp(*atd * 0.5, 0.0, duration);
            }
            // The same joins, recorded on the clip as well.  The note flags say
            // a join edit is in effect and drive the crossfade; these say the
            // recording itself runs continuously through the seam, which is
            // what lets stretchSpliceThenPitch decode the chain in one pass.
            // Two questions, so the edit-neutralising block below clears the
            // first and leaves this one -- neutralising a user's edits does not
            // make a continuous take discontinuous.
            clip.glideConnectedToNext = note.connectedToNext;
            clip.glideConnectedFromPrevious = note.connectedToPrevious;
            if (!options.preserveProjectEdits)
            {
                // Keep the project arrangement and the analysed source F0,
                // while neutralising Melodyne correction/mixer controls.  The
                // imported clip remains immediately editable without baking
                // the saved tuning, modulation, Attack, timbre or gain edits.
                note.midiNote = note.sourceMidiCenter;
                note.drift = 1.0f;
                note.modulation = 1.0f;
                note.formantSemitones = 0.0f;
                note.breath = 0.0f;
                note.gain = 1.0f;
                note.amplitudeEnvelope = { { 0.0, 0.0f },
                    { note.durationSeconds, 0.0f } };
                note.attackSpeed = 1.0f;
                note.connectedToPrevious = false;
                note.connectedToNext = false;
                clip.muted = false;
                clip.fadeInSeconds = 0.0;
                clip.fadeOutSeconds = 0.0;
                clip.crossfadeInSeconds = 0.0;
                clip.crossfadeOutSeconds = 0.0;
                clip.gain = 1.0f;
            }
            clip.notes.push_back(std::move(note));
            importedSourceUses.emplace(element,std::make_pair(source,track.clips.size()));
            track.clips.push_back(std::move(clip));
        }
        // Group only actual MUSuccessiveJoin endpoints from this import and
        // the same physical recording/source identity. Never enumerate possible
        // subsets or connect merely adjacent target/source intervals.
        std::map<std::uint32_t,std::uint32_t> usageRoots;
        for(const auto& use:importedSourceUses)usageRoots.emplace(use.first,use.first);
        const auto rootOf=[&](std::uint32_t id)
        {
            while(usageRoots.at(id)!=id)id=usageRoots.at(id);
            return id;
        };
        for(const auto& use:importedSourceUses)
            if(const auto join=graph.reference(use.first,"followingJoin"))
                if(graph.boolean(*join,"joinsPitches"))
                    if(const auto following=graph.reference(*join,"followingElement"))
                    {
                        const auto next=importedSourceUses.find(*following);
                        if(next==importedSourceUses.end()||next->second.first!=use.second.first)continue;
                        if(track.clips[use.second.second].sourceFile!=track.clips[next->second.second].sourceFile)continue;
                        const auto a=rootOf(use.first),b=rootOf(*following);
                        if(a!=b)usageRoots[b]=a;
                    }
        std::map<std::uint32_t,std::vector<std::size_t>> actualUsages;
        for(const auto& use:importedSourceUses)actualUsages[rootOf(use.first)].push_back(use.second.second);
        for(const auto& usage:actualUsages)
            if(usage.second.size()>1)
            {
                const auto id=makeId("melodyne-use");
                for(const auto index:usage.second)track.clips[index].materialSourceUsageGroupId=id;
            }
        if (!track.clips.empty()) result.project.tracks.push_back(std::move(track));
    }
    if (result.project.tracks.empty())
    { error = "Melodyne project contains no resolved playable media"; return std::nullopt; }
    if (progress) progress(1.0, "complete");
    return result;
}
}
