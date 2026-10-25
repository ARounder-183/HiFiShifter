#include "UstImporter.h"
#include "UstxImporter.h"
#include "LegacyTextCodec.h"
#include "UstText.h"

#if JUCE_WINDOWS
 // windows.h defines min and max as macros, which then eat std::max at
 // every call site below.
 #ifndef NOMINMAX
  #define NOMINMAX
 #endif
 #include <windows.h>
#endif
#include <algorithm>
#include <cmath>
#include <string>

namespace hachi::backend
{
namespace
{
// A rest in a UST is "R", and in practice also "r" and an empty lyric.  Some
// editors write a lone hyphen for one too.
bool restLyric(const juce::String& lyric)
{
    const auto trimmed = lyric.trim();
    return trimmed.isEmpty() || trimmed.equalsIgnoreCase("R")
        || trimmed == "-" || trimmed.equalsIgnoreCase("rest");
}

// UST numbers are written with a decimal point and no grouping, but a field
// may also be present and empty ("PreUtterance=") to mean "not set here".
std::optional<double> number(const juce::String& value)
{
    const auto trimmed = value.trim();
    if (trimmed.isEmpty()) return std::nullopt;
    const auto valueNumber=trimmed.getDoubleValue();
    return std::isfinite(valueNumber) ? std::optional<double>(valueNumber) : std::nullopt;
}

std::vector<double> numberList(const juce::String& value)
{
    std::vector<double> result;
    juce::StringArray parts;
    parts.addTokens(value, ",", "");
    for (const auto& part : parts)
    {
        const auto trimmed = part.trim();
        // A trailing comma is normal in a UST -- "PBY=0,0,0," -- and the empty
        // field it leaves means the note's own pitch, which is zero here.
        result.push_back(trimmed.isEmpty() ? 0.0 : trimmed.getDoubleValue());
    }
    return result;
}

bool looksLikeUtf8(const juce::MemoryBlock& bytes)
{
    const auto* data = static_cast<const unsigned char*>(bytes.getData());
    const auto size = bytes.getSize();
    for (std::size_t index = 0; index < size;)
    {
        const auto lead = data[index];
        if (lead < 0x80) { ++index; continue; }
        auto following = 0;
        if ((lead & 0xE0) == 0xC0) following = 1;
        else if ((lead & 0xF0) == 0xE0) following = 2;
        else if ((lead & 0xF8) == 0xF0) following = 3;
        else return false;
        if (index + static_cast<std::size_t>(following) >= size) return false;
        for (auto step = 1; step <= following; ++step)
            if ((data[index + static_cast<std::size_t>(step)] & 0xC0) != 0x80) return false;
        index += static_cast<std::size_t>(following) + 1;
    }
    return true;
}

#if JUCE_WINDOWS
std::wstring readIn(const char* data, int size, UINT codePage)
{
    const auto length = MultiByteToWideChar(codePage, 0, data, size, nullptr, 0);
    if (length <= 0) return {};
    std::wstring text(static_cast<std::size_t>(length), L'\0');
    MultiByteToWideChar(codePage, 0, data, size, text.data(), length);
    return text;
}

// Japanese lyrics are written in kana, and a file read in the wrong code page
// loses them: Shift-JIS kana read as GBK become rare hanzi, and GBK kana read
// as Shift-JIS become half-width katakana, which are not counted here.
int kanaCount(const std::wstring& text)
{
    return static_cast<int>(std::count_if(text.begin(), text.end(), [](wchar_t character)
    {
        return (character >= 0x3041 && character <= 0x3096)     // hiragana
            || (character >= 0x30A1 && character <= 0x30FA);    // katakana
    }));
}

juce::String codePageName(UINT codePage)
{
    if (codePage == 932) return "Shift-JIS";
    if (codePage == 936) return "GBK";
    return "code page " + juce::String(static_cast<int>(codePage));
}
#endif
}

bool UstNote::isRest() const
{
    return restLyric(lyric);
}

juce::String UstImporter::decode(const juce::MemoryBlock& bytes,
                                 juce::String& encodingUsed)
{
#if JUCE_WINDOWS
    return decode(bytes, encodingUsed, static_cast<int>(GetACP()));
#else
    return decode(bytes, encodingUsed, 0);
#endif
}

juce::String UstImporter::decode(const juce::MemoryBlock& bytes,
                                 juce::String& encodingUsed,
                                 [[maybe_unused]] int localCodePage)
{
    if (bytes.getSize() == 0)
    {
        encodingUsed = "empty";
        return {};
    }
    const auto* data = static_cast<const char*>(bytes.getData());
    if (bytes.getSize() >= 3
        && static_cast<unsigned char>(data[0]) == 0xEF
        && static_cast<unsigned char>(data[1]) == 0xBB
        && static_cast<unsigned char>(data[2]) == 0xBF)
    {
        encodingUsed = "UTF-8";
        return juce::String::fromUTF8(data + 3, static_cast<int>(bytes.getSize()) - 3);
    }
    if (looksLikeUtf8(bytes))
    {
        encodingUsed = "UTF-8";
        return juce::String::fromUTF8(data, static_cast<int>(bytes.getSize()));
    }
#if JUCE_WINDOWS
    // UTAU reads a UST in the code page of the machine running it, and so
    // did this -- which garbled every Japanese file opened on a Chinese
    // machine.  So Shift-JIS is read beside the local code page, and GBK too
    // where the local one is Shift-JIS or GBK, and the reading with the most
    // kana is taken.  A file with no kana either way reads as it always did,
    // in the local code page.  GBK is not tried elsewhere: Big5 puts common
    // hanzi on the bytes GBK uses for hiragana.
    const auto size = static_cast<int>(bytes.getSize());
    const auto local = static_cast<UINT>(localCodePage);
    std::vector<UINT> candidates { local };
    const auto consider = [&candidates](UINT codePage)
    {
        if (std::find(candidates.begin(), candidates.end(), codePage) == candidates.end())
            candidates.push_back(codePage);
    };
    consider(932);
    if (local == 932 || local == 936) consider(936);
    std::wstring best;
    auto bestKana = -1;
    auto bestPage = local;
    for (const auto codePage : candidates)
    {
        auto reading = readIn(data, size, codePage);
        const auto kana = kanaCount(reading);
        if (!reading.empty() && kana > bestKana)
        {
            best = std::move(reading);
            bestKana = kana;
            bestPage = codePage;
        }
    }
    if (!best.empty())
    {
        encodingUsed = codePageName(bestPage);
        return juce::String(best.c_str(), best.size());
    }
#endif
    encodingUsed = "Latin-1";
    return juce::String::createStringFromData(data, static_cast<int>(bytes.getSize()));
}

UstProject UstImporter::parse(const juce::String& text, juce::StringArray& warnings)
{
    UstProject project;
    project.sourceText=text;
    const auto sections=usttext::sections(text);
    int sectionIndex=-1; bool explicitMode=false, hasMode1=false;
    juce::StringArray lines;
    lines.addLines(text);

    enum class Section { none, version, setting, note, end };
    auto section = Section::none;
    UstNote current;
    auto haveNote = false;
    auto malformedEnvelopes = 0;

    const auto finishNote = [&]
    {
        if (haveNote) project.notes.push_back(current);
        current = UstNote{};
        haveNote = false;
    };

    for (const auto& raw : lines)
    {
        const auto line = raw.trim();
        if (line.isEmpty()) continue;
        if (line.startsWithChar('[') && line.endsWithChar(']'))
        {
            finishNote();
            ++sectionIndex;
            const auto tag = line.substring(1, line.length() - 1);
            if (tag.equalsIgnoreCase("#VERSION")) section = Section::version;
            else if (tag.equalsIgnoreCase("#SETTING")) section = Section::setting;
            else if (tag.equalsIgnoreCase("#TRACKEND")) section = Section::end;
            else if (usttext::noteTag(tag))
            {
                // Context and vendor sections remain opaque, never audible notes.
                current.sourceSectionIndex=sectionIndex;
                if(sectionIndex>=0 && (size_t)sectionIndex<sections.size()) current.sourceSection=sections[(size_t)sectionIndex].text;
                section = Section::note;
                haveNote = true;
            }
            else section = Section::none;
            continue;
        }
        if (section == Section::end) break;
        const auto split = line.indexOfChar('=');
        if (split < 0) continue;
        const auto key = line.substring(0, split).trim();
        const auto value = line.substring(split + 1).trim();

        if (section == Section::setting)
        {
            if (key.equalsIgnoreCase("Tempo"))
            {
                if (const auto tempo = number(value)) project.tempo = *tempo;
            }
            else if (key.equalsIgnoreCase("ProjectName")) project.name = value;
            else if (key.equalsIgnoreCase("VoiceDir")) project.voiceDirectory = value;
            else if (key.equalsIgnoreCase("Flags")) project.globalFlags = value;
            else if (key.equalsIgnoreCase("Mode2")) {explicitMode=true;project.mode2=value.equalsIgnoreCase("True") || value=="1";}
            continue;
        }
        if (section != Section::note || !haveNote) continue;

        if (key.equalsIgnoreCase("Length"))
        {
            if (const auto ticks = number(value))
                current.lengthTicks = std::max(0, static_cast<int>(std::lround(*ticks)));
        }
        else if (key.equalsIgnoreCase("Lyric")) current.lyric = value;
        else if (key.equalsIgnoreCase("NoteNum"))
        {
            if (const auto note = number(value))
                current.noteNum = juce::jlimit(0, 127, static_cast<int>(std::lround(*note)));
        }
        else if (key.equalsIgnoreCase("Tempo")) current.tempo = number(value);
        else if (key.equalsIgnoreCase("PreUtterance")) current.preutteranceMs = number(value);
        else if (key.equalsIgnoreCase("VoiceOverlap")) current.overlapMs = number(value);
        else if (key.equalsIgnoreCase("Velocity"))
        {
            if (const auto velocity = number(value))
                current.velocity = static_cast<int>(std::lround(*velocity));
        }
        else if (key.equalsIgnoreCase("Intensity"))
        {
            if (const auto intensity = number(value))
                current.intensity = static_cast<int>(std::lround(*intensity));
        }
        else if (key.equalsIgnoreCase("Modulation")) current.modulation=number(value);
        else if (key.equalsIgnoreCase("STP")) current.stpMs=number(value);
        else if (key.equalsIgnoreCase("PBStart")) current.mode1StartMs=number(value).value_or(0);
        else if (key.equalsIgnoreCase("PitchBend") || key.equalsIgnoreCase("Pitches")) { current.mode1Cents=numberList(value); hasMode1=hasMode1 || !current.mode1Cents.empty(); }
        else if (key.equalsIgnoreCase("Flags")) current.flags = value;
        else if (key.equalsIgnoreCase("PBS"))
        {
            // "offset;pitch", and also just "offset" -- a bend that starts at
            // the note's own pitch leaves the second field off entirely.
            const auto semicolon = value.indexOfChar(';');
            const auto offset = semicolon >= 0 ? value.substring(0, semicolon) : value;
            const auto pitch = semicolon >= 0 ? value.substring(semicolon + 1) : juce::String();
            current.pitchStartMs = number(offset).value_or(0.0);
            current.pitchStartTenths = number(pitch).value_or(0.0);
            current.hasPitchBend = true;
        }
        else if (key.equalsIgnoreCase("PBW"))
        {
            current.widthsMs = numberList(value);
            current.hasPitchBend = true;
        }
        else if (key.equalsIgnoreCase("PBY")) current.pitchTenths = numberList(value);
        else if (key.equalsIgnoreCase("PBM"))
        {
            current.shapes.clear();
            current.shapes.addTokens(value, ",", "");
            for (auto& shape : current.shapes) shape = shape.trim();
        }
        else if (key.equalsIgnoreCase("VBR"))
        {
            const auto fields = numberList(value);
            const auto at = [&fields](std::size_t index)
            {
                return index < fields.size() ? fields[index] : 0.0;
            };
            // Three numbers are the least that says anything: how long, how
            // fast, how deep.  A note UTAU left without a vibrato writes a
            // length of zero, and so does one whose vibrato was taken off.
            if (fields.size() >= 3)
            {
                current.vibratoLengthPercent = at(0);
                current.vibratoCycleMs = at(1);
                current.vibratoDepthCents = at(2);
                current.vibratoFadeInPercent = at(3);
                current.vibratoFadeOutPercent = at(4);
                current.vibratoPhasePercent = at(5);
                current.vibratoOffsetPercent = at(6);
                current.hasVibrato = current.vibratoLengthPercent > 0.0
                    && current.vibratoDepthCents != 0.0
                    && current.vibratoCycleMs > 0.0;
            }
        }
        else if (key.equalsIgnoreCase("Envelope"))
        {
            juce::StringArray fields;
            fields.addTokens(value, ",", "");
            for (auto& field : fields) field = field.trim();
            // Seven numbers are the whole of an older envelope; "%" separates
            // the two that were added later, and either of those may be left
            // off again.  Read what is there and no more.
            if (fields.size() >= 7)
            {
                const auto at = [&fields](int index)
                {
                    return index < fields.size() ? number(fields[index]) : std::nullopt;
                };
                current.hasEnvelope = true;
                current.envelopeP1 = at(0).value_or(0.0);
                current.envelopeP2 = at(1).value_or(0.0);
                current.envelopeP3 = at(2).value_or(0.0);
                current.envelopeV1 = at(3).value_or(0.0);
                current.envelopeV2 = at(4).value_or(100.0);
                current.envelopeV3 = at(5).value_or(100.0);
                current.envelopeV4 = at(6).value_or(0.0);
                // The marker is normally at index 7, but a file that omits it
                // still counts the same fields after it.
                auto tail = 7;
                if (tail < fields.size() && fields[tail] == "%") ++tail;
                if (const auto p4 = at(tail)) current.envelopeP4 = *p4;
                const auto p5 = at(tail + 1);
                const auto v5 = at(tail + 2);
                // A p5 with no v5 says where a point is but not what it is, so
                // it places nothing rather than a guessed level.
                if (p5 && v5)
                {
                    current.hasMiddlePoint = true;
                    current.envelopeP5 = *p5;
                    current.envelopeV5 = *v5;
                }
            }
            else ++malformedEnvelopes;
        }
    }
    finishNote();

    if(!explicitMode && hasMode1)project.mode2=false;
    for(auto& note:project.notes)note.mode2=project.mode2;
    if (project.notes.empty())
        warnings.add("no notes found");
    if (malformedEnvelopes > 0)
        warnings.add(juce::String(malformedEnvelopes)
                     + " note(s) have an envelope with too few values to read");
    return project;
}

std::optional<UstProject> UstImporter::read(const juce::File& file, juce::String& error,
                                            juce::StringArray& warnings, int encodingOverride)
{
    if(file.hasFileExtension("ustx"))return UstxImporter::read(file,error,warnings);
    const auto document=LegacyTextCodec::read(file,error,encodingOverride,false);
    if(!document)return std::nullopt;
    const auto& text=document->text;
    const auto& encoding=document->encoding;
    if (text.isEmpty())
    {
        error = "Empty UST file: " + file.getFullPathName();
        return std::nullopt;
    }
    auto project = parse(text, warnings);
    project.sourceEncoding=document->encoding; project.sourceBom=document->bom; project.sourceBytes=document->bytes.toBase64Encoding();
    if (project.notes.empty())
    {
        error = "No notes in " + file.getFullPathName();
        return std::nullopt;
    }
    if (encoding != "UTF-8")
        warnings.add("read as " + encoding);
    if (project.name.isEmpty()) project.name = file.getFileNameWithoutExtension();
    return project;
}
}
