#include "backend/MixedEnvelopeIO.h"
#include "backend/CvvcPhonemeCache.h"
#include "NativeAudioTrim.h"
#include "NativeNoteJoin.h"
#include "NativeAudioOverlap.h"
#include "NativeAudioLink.h"
#include "NativeAudioDisconnect.h"
#include "NativeUnpitchedRegions.h"
#include "DiffSingerParameterCurves.h"
#include "DiffSingerPitchHandles.h"
#include "ProjectModel.h"
#include "backend/UstExchange.h"
#include "ProjectFileIO.h"
#include "HamoodProject.h"
#include "TrackGainEnvelope.h"
#include "ClipParts.h"
#include "Pinyin.h"
#include "SampleSettings.h"
#include "OttoMaterialIO.h"
#include <juce_cryptography/juce_cryptography.h>
#include "backend/UstImporter.h"
#include "backend/UtauRenderer.h"
#include "backend/AmplitudeEnvelopeCurve.h"
#include "backend/UtauMixEnvelope.h"
#include "backend/NsfHifiganRenderer.h"
#include "backend/AnalysisService.h"
#include "NativeSourceTimeMap.h"
#include "NativeNoteTiming.h"
#include "NativePitchIdentity.h"
#include "NativeAudioClipboard.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <optional>
#include <set>

namespace hachi
{
juce::String trackTypeName(TrackType type)
{
    switch(type){case TrackType::hachi:return "hachi";case TrackType::utau:return "utau";case TrackType::diffSinger:return "diffsinger";}return "hachi";
}
std::optional<TrackType> parseTrackType(const juce::String& name)
{
    const auto value=name.trim().toLowerCase();
    if(value=="hachi"||value=="hachishifter")return TrackType::hachi;
    if(value=="utau")return TrackType::utau;
    if(value=="diffsinger"||value=="diff-singer")return TrackType::diffSinger;
    return {};
}
TrackType effectiveTrackType(const TrackData& track)
{
    if(track.trackType)return *track.trackType;
    // Compatibility for raw legacy DTOs only. Model adoption/load freezes this
    // result, so later voicebank/filesystem changes cannot reinterpret a track.
    if(track.accompaniment)return TrackType::hachi;
    if(track.pitchAlgorithm==PitchAlgorithm::utau)
        return track.utauMode==UtauMode::mou&&track.voicebankDirectory.getChildFile("dsconfig.yaml").existsAsFile()?TrackType::diffSinger:TrackType::utau;
    const auto legacyBank=track.pitchAlgorithm==PitchAlgorithm::nsfHifigan&&!track.nativeNsfAudio
        &&(utauModeUsesRegions(track.utauMode)||track.voicebankDirectory.isDirectory())
        &&!track.voicebankDirectory.getChildFile("dsconfig.yaml").existsAsFile();
    return legacyBank&&(track.outputEngine==UtauOutputEngine::inherit||track.outputEngine==UtauOutputEngine::pcNsfHifigan)?TrackType::utau:TrackType::hachi;
}
namespace
{
void configureNewTrackType(TrackData& track,TrackType type)
{
    track.trackType=type;
    if(type==TrackType::hachi){track.pitchAlgorithm=PitchAlgorithm::nsfHifigan;track.nativeNsfAudio=true;track.renderOrder=RenderOrder::stretchSpliceThenPitch;}
    else{track.pitchAlgorithm=PitchAlgorithm::utau;track.nativeNsfAudio=false;track.utauMode=type==TrackType::diffSinger?UtauMode::mou:UtauMode::classic;}
}
}
PitchAlgorithm defaultPitchAlgorithm(const juce::File& modelDirectory)
{
    // A missing model is a configuration error, not an implicit engine switch.
    juce::ignoreUnused(modelDirectory);
    return PitchAlgorithm::nsfHifigan;
}
juce::String nativeSegmentRoleName(NativeSegmentRole role)
{
    switch (role)
    {
        case NativeSegmentRole::consonant: return "consonant";
        case NativeSegmentRole::vowel: return "vowel";
        case NativeSegmentRole::transition: return "transition";
        case NativeSegmentRole::silence: return "silence";
        case NativeSegmentRole::breath: return "breath";
        case NativeSegmentRole::noise: return "noise";
        case NativeSegmentRole::ending: return "ending";
        case NativeSegmentRole::unknown: break;
    }
    return "unknown";
}

NativeSegmentRole parseNativeSegmentRole(const juce::String& value)
{
    const auto role = value.trim().toLowerCase();
    if (role == "consonant") return NativeSegmentRole::consonant;
    if (role == "vowel") return NativeSegmentRole::vowel;
    if (role == "transition") return NativeSegmentRole::transition;
    if (role == "silence") return NativeSegmentRole::silence;
    if (role == "breath") return NativeSegmentRole::breath;
    if (role == "noise") return NativeSegmentRole::noise;
    if (role == "ending") return NativeSegmentRole::ending;
    return NativeSegmentRole::unknown;
}

float renderedPitchCents(const NoteData& note, const PitchPoint& point)
{
    if (point.hasManualTarget) return point.manualTargetCents;

    // Melodyne stores an exact zero for notes flattened with the pitch
    // modulation tool.  Some analysed source points do not contain a separate
    // pitchWithoutVibrato curve (it is byte-for-byte equal to pitchCent), so
    // the generic drift/modulation decomposition would otherwise leave the
    // original contour untouched.  Treat the saved zero as the authoritative
    // flat-note edit for both display and rendering.
    if (note.modulation <= 1.0e-4f) return 0.0f;

    return note.drift * point.withoutVibratoCents
        + note.modulation * (point.relativeCents - point.withoutVibratoCents);
}

float shapedSegmentProgress(PitchCurveShape shape, float u,
                            float bezierX1, float bezierY1,
                            float bezierX2, float bezierY2)
{
    u = juce::jlimit(0.0f, 1.0f, u);
    switch (shape)
    {
        case PitchCurveShape::smooth: return u * u * (3.0f - 2.0f * u);
        case PitchCurveShape::easeIn: return u * u;
        case PitchCurveShape::easeOut: return 1.0f - (1.0f - u) * (1.0f - u);
        case PitchCurveShape::customBezier: break;
        // A contour-wide shape has no per-segment answer: whoever owns a whole
        // contour handles that one, and everything else is a straight line.
        case PitchCurveShape::natural:
        case PitchCurveShape::linear:
        default: return u;
    }
    // Invert the Bezier X component to turn the normalised timeline U into the
    // curve parameter T, then evaluate Y.  Newton iteration is fast for
    // ordinary handles; bisection is the fallback around flat derivatives.
    const auto x1 = juce::jlimit(0.0f, 1.0f, bezierX1);
    const auto x2 = juce::jlimit(0.0f, 1.0f, bezierX2);
    const auto cubic = [](float t, float first, float second)
    {
        const auto inverse = 1.0f - t;
        return 3.0f * inverse * inverse * t * first
            + 3.0f * inverse * t * t * second + t * t * t;
    };
    const auto derivative = [](float t, float first, float second)
    {
        const auto inverse = 1.0f - t;
        return 3.0f * inverse * inverse * first
            + 6.0f * inverse * t * (second - first)
            + 3.0f * t * t * (1.0f - second);
    };
    auto parameter = u;
    for (auto iteration = 0; iteration < 6; ++iteration)
    {
        const auto slope = derivative(parameter, x1, x2);
        if (std::abs(slope) < 1.0e-5f) break;
        const auto candidate = parameter - (cubic(parameter, x1, x2) - u) / slope;
        if (candidate < 0.0f || candidate > 1.0f) break;
        parameter = candidate;
    }
    auto lower = 0.0f;
    auto upper = 1.0f;
    for (auto iteration = 0; iteration < 18; ++iteration)
    {
        if (cubic(parameter, x1, x2) < u) lower = parameter;
        else upper = parameter;
        parameter = (lower + upper) * 0.5f;
    }
    return cubic(parameter, bezierY1, bezierY2);
}

juce::String utauModeLabel(UtauMode mode)
{
    return mode == UtauMode::jie ? juce::String::fromUTF8("界•UTAU")
         : mode == UtauMode::mou ? juce::String::fromUTF8("谋•UTAU")
                                 : juce::String("UTAU");
}

juce::String utauModeKey(UtauMode mode)
{
    // "utau4" is what 界 has always been called in project files and over
    // MCP, so it stays exactly that; only the new mode needs a new word.
    return mode == UtauMode::jie ? juce::String("utau4")
         : mode == UtauMode::mou ? juce::String("utaumou")
                                 : juce::String("utau");
}

int utauModePickerItem(UtauMode mode)
{
    return mode == UtauMode::jie ? 8 : mode == UtauMode::mou ? 9 : 7;
}

std::optional<UtauMode> utauModeForPickerItem(int itemId)
{
    switch (itemId)
    {
        case 7: return UtauMode::classic;
        case 8: return UtauMode::jie;
        case 9: return UtauMode::mou;
        default: return std::nullopt;
    }
}

float scaledEnvelopeGainDb(float gainDb, double factor)
{
    // Silence is silence however it is scaled, and the renderer reads anything
    // at or below -60 dB as nothing at all.
    if (gainDb <= -59.9f || factor <= 1.0e-9) return -60.0f;
    const auto level = std::pow(10.0, gainDb / 20.0) * factor;
    if (level <= 1.0e-4) return -60.0f;
    // The same ceiling the UST importer writes envelopes against: +12 dB is
    // 400%, which is as far as a base of 200 can take a point already at 200%.
    return juce::jlimit(-60.0f, 12.0f, static_cast<float>(20.0 * std::log10(level)));
}

std::vector<AmplitudeEnvelopePoint> scaledAmplitudeEnvelope(
    const std::vector<AmplitudeEnvelopePoint>& points, float basePercent)
{
    auto scaled = points;
    const auto factor = juce::jlimit(0.0f, 200.0f, basePercent) / 100.0;
    if (std::abs(factor - 1.0) < 1.0e-9) return scaled;
    for (auto& point : scaled) point.gainDb = scaledEnvelopeGainDb(point.gainDb, factor);
    return scaled;
}

std::vector<AmplitudeEnvelopePoint> unscaledAmplitudeEnvelope(
    const std::vector<AmplitudeEnvelopePoint>& points, float basePercent)
{
    auto plain = points;
    const auto factor = juce::jlimit(0.0f, 200.0f, basePercent) / 100.0;
    if (std::abs(factor - 1.0) < 1.0e-9 || factor <= 1.0e-9) return plain;
    for (auto& point : plain) point.gainDb = scaledEnvelopeGainDb(point.gainDb, 1.0 / factor);
    return plain;
}

UtauMode parseUtauMode(const juce::String& text)
{
    const auto value = text.trim().toLowerCase();
    if (value == "utau4" || value == "jie") return UtauMode::jie;
    if (value == "utaumou" || value == "mou") return UtauMode::mou;
    return UtauMode::classic;
}

const std::vector<FlagCurveKind>& flagCurveKinds()
{
    // Ordered as they are offered: the formant shift first, since it is what a
    // curve is usually wanted for, then the rest of the timbre controls.
    static const std::vector<FlagCurveKind> kinds {
        { "g",  "共振峰平移", -50.0f,  50.0f },
        { "Mt", "张力",      -100.0f, 100.0f },
        { "Rd", "声门 Rd",    -100.0f, 100.0f },
        { "Mo", "开口度",     -100.0f, 100.0f },
        { "ME", "共振峰强调", -100.0f, 100.0f },
        { "Mr", "歌手共振峰", -100.0f, 100.0f },
        { "MH", "高频滚降",   -100.0f, 100.0f },
        { "Mq", "高次谐波滚降", 0.0f,  100.0f },
        { "Mf", "共振峰调谐",   0.0f,  100.0f },
        { "Mb", "元音气声",   -100.0f, 100.0f },
        { "Ab", "全帧气声",   -100.0f, 100.0f },
        { "Md", "干燥度",     -100.0f, 100.0f },
        { "Mn", "噪声平滑",   -100.0f, 100.0f },
        { "NA", "鼻音度",     -100.0f, 100.0f },
        { "RG", "自动混声",   -100.0f, 100.0f },
        { "b",  "清辅音噪声",  -20.0f, 100.0f, true },
        { "bh", "辅音区谐波",  -20.0f, 100.0f, true },
        // 嘶吼 is not a frame parameter like the rest of these: the engine runs
        // it on the finished mix and reads this curve for its depth, sample by
        // sample.  Drawn in the same lane all the same -- one way to shape a
        // flag, whichever end of the engine reads it.
        { "MY", "嘶吼",          0.0f, 100.0f }
    };
    return kinds;
}

const std::vector<FlagCurveKind>& hifisamplerFlagCurveKinds()
{
    // Namespace avoids changing the ranges/units of saved WCSNDM curves.
    static const std::vector<FlagCurveKind> kinds {
        {"HIFI:g", "共振峰 / 性别", -600, 600},
        {"HIFI:Hb", "气声 / 噪声", 0, 500, false, 100},
        {"HIFI:Hv", "谐波 / 实声", 0, 150, false, 100},
        {"HIFI:Ht", "张力", -100, 100},
        {"HIFI:HG", "嘶吼 / 沙哑", 0, 100},
        {"HIFI:P", "响度归一化强度", 0, 100, false, 100},
        {"HIFI:t", "音高偏移（音分）", -1200, 1200},
        {"HIFI:A", "随音高变化调制振幅", -100, 100}
    };
    return kinds;
}

const std::vector<FlagCurveKind>& diffSingerFlagCurveKinds()
{
    static const std::vector<FlagCurveKind> kinds {
        { "DS:DYN", "响度（0.1 dB）", -240, 120 },
        { "DS:GENC", "共振峰 / 性别", -100, 100 },
        { "DS:VELC", "发音速度", 0, 200, false, 100 },
        { "DS:ENE", "能量偏移", -100, 100 },
        { "DS:BREC", "气声偏移", -100, 100 },
        { "DS:TENC", "张力偏移", -100, 100 },
        { "DS:VOIC", "实声偏移（100 为零偏移）", 0, 100, false, 100 },
        { "DS:PEXP", "音高表现（生成 pitch 时）", 0, 100, false, 100 },
        { "DS:SHFC", "音区偏移（音分）", -1200, 1200 }
    };
    return kinds;
}

const std::vector<FlagCurveKind>& diffSingerParameterKinds()
{
    static const std::vector<FlagCurveKind> kinds {
        {"DS:ABS:ENE", "能量实参（dB）", -96, 0, false, -40},
        {"DS:ABS:BREC", "气声实参（dB）", -96, 0, false, -40},
        {"DS:ABS:TENC", "张力实参（模型单位）", -10, 10},
        {"DS:ABS:VOIC", "实声量实参（dB）", -96, 0, false, -40}
    };
    return kinds;
}

const FlagCurveKind& flagCurveKindFor(const juce::String& flag)
{
    for (const auto& kind : hifisamplerFlagCurveKinds())
        if (flag == kind.flag) return kind;
    for (const auto& kind : diffSingerParameterKinds())
        if (isDiffSingerParameter(flag) && flag.fromLastOccurrenceOf(":", false, false)
            == juce::String(kind.flag).fromLastOccurrenceOf(":", false, false)) return kind;
    for (const auto& kind : diffSingerFlagCurveKinds())
        if (flag == kind.flag) return kind;
    static const FlagCurveKind colour { "DS:CLR:", "音色混合权重（%）", 0, 100 };
    if (flag.startsWith("DS:CLR:")) return colour;
    for (const auto& kind : flagCurveKinds())
        if (flag == kind.flag) return kind;
    return flagCurveKinds().front();
}

std::vector<FlagCurvePoint> flagCurvePointsFor(const NoteData& note,
                                               const juce::String& flag)
{
    for (const auto& curve : note.utauFlagCurves)
        if (curve.flag == flag) return curve.points;
    if (flag.startsWith("HIFI:")) return flagCurvePointsFor(note, flag.substring(5));
    if (flag.startsWith("DS:ABS:")) return flagCurvePointsFor(note, "DS:REF:"+flag.substring(7));
    if (flag.startsWith("DS:REF:")) return flagCurvePointsFor(note, "DS:AUTO:"+flag.substring(7));
    return {};
}

std::vector<std::pair<juce::String, std::vector<std::pair<double, double>>>>
sampleDiffSingerFlagCurves(const NoteData& note)
{
    std::vector<std::pair<juce::String, std::vector<std::pair<double, double>>>> result;
    auto all = note.utauFlagCurves;
    for (const auto& kind : diffSingerParameterKinds()) {
        // Automatic snapshots are display/persistence metadata, not frozen synthesis inputs.
        const auto code = juce::String(kind.flag).substring(7);
        const auto explicitBase = std::any_of(note.utauFlagCurves.begin(), note.utauFlagCurves.end(),
            [&](const auto& c) { return !c.points.empty() && (c.flag == kind.flag || c.flag == "DS:REF:"+code); });
        const auto base = explicitBase ? flagCurvePointsFor(note, kind.flag) : std::vector<FlagCurvePoint>{};
        if (!base.empty()) all.push_back({kind.flag, base});
    }
    std::set<juce::String> emitted;
    for (const auto& curve : all)
    {
        if (!curve.flag.startsWith("DS:") || curve.points.empty()) continue;
        if (curve.flag.startsWith("DS:REF:") || curve.flag.startsWith("DS:PTS:") || curve.flag.startsWith("DS:AUTO:")) continue;
        if (!note.utauFlagCurveEnabled && !curve.flag.startsWith("DS:ABS:")) continue;
        if (!emitted.insert(curve.flag).second) continue;
        std::vector<std::pair<double, double>> sampled;
        for (std::size_t i = 0; i < curve.points.size(); ++i)
        {
            const auto& point = curve.points[i];
            if (i > 0 && point.shape != PitchCurveShape::linear)
            {
                const auto from = curve.points[i-1].timeSeconds;
                const auto steps = juce::jlimit(2, 6500, static_cast<int>(std::ceil((point.timeSeconds-from)/.005)));
                for (int step = 1; step < steps; ++step)
                {
                    const auto time = from + (point.timeSeconds-from)*step/steps;
                    sampled.emplace_back(time, flagCurveValueAt(curve.points, time));
                }
            }
            sampled.emplace_back(point.timeSeconds, point.value);
        }
        result.emplace_back(curve.flag, std::move(sampled));
    }
    return result;
}

float flagCurveValueAt(const std::vector<FlagCurvePoint>& points, double timeSeconds)
{
    if (points.empty()) return 0.0f;
    if (timeSeconds <= points.front().timeSeconds) return points.front().value;
    if (timeSeconds >= points.back().timeSeconds) return points.back().value;
    const auto right = std::upper_bound(points.begin(), points.end(), timeSeconds,
        [](double value, const FlagCurvePoint& point)
        {
            return value < point.timeSeconds;
        });
    if (right == points.begin()) return points.front().value;
    if (right == points.end()) return points.back().value;
    const auto& next = *right;
    const auto& left = *std::prev(right);
    const auto span = next.timeSeconds - left.timeSeconds;
    if (span <= 1.0e-9) return next.value;
    const auto u = static_cast<float>((timeSeconds - left.timeSeconds) / span);
    // The shape belongs to the segment arriving at a point, as on the pitch
    // line, so the first point never has its own shape consulted.
    const auto shaped = shapedSegmentProgress(next.shape, u, next.bezierX1,
                                              next.bezierY1, next.bezierX2,
                                              next.bezierY2);
    return left.value + (next.value - left.value) * shaped;
}

float diffSingerPitchOffsetAt(const NoteData& note, double time)
{
    return note.diffSingerPitchOffset.empty() ? 0.0f : evaluatePitchCurve(note.diffSingerPitchOffset, time);
}

std::vector<PitchCurveEditPoint> ownPitchPoints(const NoteData& note)
{
    if (!note.pitchControlPoints.empty()) return note.pitchControlPoints;
    // The contour as the renderer reads it: every point, joined straight.
    std::vector<PitchCurveEditPoint> points;
    points.reserve(note.contour.size());
    for (const auto& point : note.contour)
    {
        PitchCurveEditPoint own { point.timeSeconds,
                                  note.midiNote + renderedPitchCents(note, point) / 100.0f };
        own.shape = PitchCurveShape::linear;
        points.push_back(own);
    }
    if (points.empty())
        points = { { 0.0, note.midiNote }, { note.durationSeconds, note.midiNote } };
    std::stable_sort(points.begin(), points.end(), [](const auto& left, const auto& right)
    {
        return left.timeSeconds < right.timeSeconds;
    });
    return points;
}

float SharedPitchLine::midiAt(double absoluteSeconds) const
{
    if (pieces.empty()) return 60.0f;
    auto piece = pieces.begin();
    for (auto next = std::next(pieces.begin()); next != pieces.end(); ++next)
    {
        if (next->from > absoluteSeconds) break;
        piece = next;
    }
    return evaluatePitchCurve(piece->points, absoluteSeconds);
}

std::vector<double> SharedPitchLine::cornersBetween(double from, double to) const
{
    std::vector<double> corners { from };
    for (std::size_t index = 0; index < pieces.size(); ++index)
    {
        const auto start = index == 0 ? -std::numeric_limits<double>::infinity()
                                      : pieces[index].from;
        const auto end = index + 1 < pieces.size() ? pieces[index + 1].from
                                                   : std::numeric_limits<double>::infinity();
        if (index > 0 && start > from && start < to) corners.push_back(start);
        for (const auto& point : pieces[index].points)
            if (point.timeSeconds > std::max(from, start) && point.timeSeconds < std::min(to, end))
                corners.push_back(point.timeSeconds);
    }
    corners.push_back(to);
    std::sort(corners.begin(), corners.end());
    corners.erase(std::unique(corners.begin(), corners.end(), [](double left, double right)
    {
        return std::abs(left - right) < 1.0e-9;
    }), corners.end());
    return corners;
}

static SharedPitchLines nativeSharedPitchLines(const TrackData& track,
    const juce::String& replacedNoteId, const std::vector<PitchCurveEditPoint>* replacedPoints,
    bool envelopeGrouping)
{
    struct Entry
    {
        juce::String id;
        double start, end;
        bool edited;
        bool independent;
        bool handlesPlaced;
        bool scoreLocked;
        std::vector<PitchCurveEditPoint> points;
    };
    std::map<juce::String, std::vector<Entry>> groups;
    for (const auto& parent : track.clips)
        for (const auto& clip : expandedClipParts(parent))
            for (const auto& note : clip.notes)
            {
                if (note.nativeUnpitched) continue;
                const auto replaced = replacedPoints && note.id == replacedNoteId && !replacedPoints->empty();
                Entry entry {note.id, clip.startSeconds + note.startSeconds,
                    clip.startSeconds + note.startSeconds + note.durationSeconds,
                    replaced || note.materialTargetPitchLocked || !nativePitchIsUnedited(note), note.nativeIndependentPitch && !envelopeGrouping,
                    replaced || note.nativePitchHandlesPlaced, note.materialTargetPitchLocked,
                    replaced ? *replacedPoints : ownPitchPoints(note)};
                for (auto& point : entry.points) point.timeSeconds += entry.start;
                const auto group = clip.nativePitchGroupId.isNotEmpty() ? clip.nativePitchGroupId
                    : parent.nativeAudioLinked ? parent.id : clip.id;
                groups[group].push_back(std::move(entry));
            }
    SharedPitchLines result;
    for (auto& [group, entries] : groups)
    {
        juce::ignoreUnused(group);
        std::stable_sort(entries.begin(), entries.end(), [](const auto& a, const auto& b)
        { return a.start < b.start; });
        for (std::size_t first = 0; first < entries.size();)
        {
            auto last = first;
            // Independent overlapping voices cannot be collapsed into one F0.
            while (last + 1 < entries.size() && !entries[last].independent && !entries[last + 1].independent
                && entries[last + 1].start >= entries[last].end - .002)
                ++last;
            if (last == first) { ++first; continue; }
            struct OwnedPoint { PitchCurveEditPoint point; std::size_t owner; };
            std::vector<OwnedPoint> points;
            bool edited = false;
            for (auto i = first; i <= last; ++i)
            {
                const auto& entry = entries[i]; edited |= entry.edited;
                for (std::size_t j = 0; j < entry.points.size(); ++j)
                {
                    const auto& point = entry.points[j];
                    // The internal cut is not a second note ending. The next
                    // section supplies its one joint point; real interior edits
                    // and the outermost endpoints remain independently editable.
                    if (i < last && j + 1 == entry.points.size()
                        && std::abs(point.timeSeconds - entry.end) < 1.0e-7) continue;
                    points.push_back({point, i});
                }
            }
            std::stable_sort(points.begin(), points.end(), [](const auto& a, const auto& b)
            { return a.point.timeSeconds < b.point.timeSeconds; });
            std::vector<OwnedPoint> unique;
            for (const auto& point : points)
            {
                if (!unique.empty() && std::abs(unique.back().point.timeSeconds - point.point.timeSeconds) < 1.0e-7)
                    unique.back() = point;
                else unique.push_back(point);
            }
            if (unique.empty()) { first = last + 1; continue; }
            auto line = std::make_shared<SharedPitchLine>();
            SharedPitchLine::Piece piece; piece.from = -std::numeric_limits<double>::infinity();
            for (const auto& point : unique) piece.points.push_back(point.point);
            line->pieces.push_back(std::move(piece));
            for (auto i = first; i <= last; ++i)
            {
                SharedPitchLineMember member; member.line = line;
                member.nativeSharedCurve = true; member.renderSharedCurve = edited;
                member.joinsPrevious = i > first; member.joinsNext = i < last;
                member.drawFrom = i == first ? std::min(entries[i].start, unique.front().point.timeSeconds) : entries[i].start;
                member.drawTo = i == last ? std::max(entries[i].end, unique.back().point.timeSeconds) : entries[i + 1].start;
                for (const auto& point : unique) if (point.owner == i) member.nativeHandleTimes.push_back(point.point.timeSeconds);
                result.byNote[entries[i].id] = std::move(member);
            }
            first = last + 1;
        }
        if (envelopeGrouping) continue;
        // Keep each independently edited body as its own interpolation piece.
        // Join touching bodies only in a one-control-point-unit window; never
        // feed their endpoints into a single whole-phrase natural spline.
        const auto ensureMember = [&](const Entry& entry) -> SharedPitchLineMember&
        {
            if (auto found = result.byNote.find(entry.id); found != result.byNote.end()) return found->second;
            auto line = std::make_shared<SharedPitchLine>();
            line->pieces.push_back({-std::numeric_limits<double>::infinity(), entry.points});
            SharedPitchLineMember member;
            member.line = line; member.nativeSharedCurve = true; member.renderSharedCurve = entry.edited;
            member.drawFrom = entry.points.empty() ? entry.start : std::min(entry.start,entry.points.front().timeSeconds);
            member.drawTo = entry.points.empty() ? entry.end : std::max(entry.end,entry.points.back().timeSeconds);
            for (const auto& point : entry.points) member.nativeHandleTimes.push_back(point.timeSeconds);
            return result.byNote.emplace(entry.id, std::move(member)).first->second;
        };
        for(const auto& entry:entries)if(entry.scoreLocked)ensureMember(entry);
        for (std::size_t i = 1; i < entries.size(); ++i)
        {
            const auto& left = entries[i - 1]; const auto& right = entries[i];
            if (!(left.independent || right.independent) || std::abs(left.end - right.start) > 1.e-7) continue;
            if (!left.edited && !right.edited) continue; // Restoring source F0 is not a new glide edit.
            // Overlapping voices (including a third voice crossing the seam)
            // are independent even if two of their edges happen to touch.
            const auto seam = right.start;
            if (std::any_of(entries.begin(), entries.end(), [&](const auto& e)
                { return e.id != left.id && e.id != right.id && e.start < seam && e.end > seam; })) continue;
            const auto half = std::min({.0005, (left.end-left.start)*.25, (right.end-right.start)*.25});
            if (half <= 1.e-9) continue;
            // Only seed the original 1 ms bridge for automatic handles.
            // Placed endpoints themselves delimit the bridge; never recreate
            // a handle at the audio boundary after a horizontal drag.
            const auto from = left.independent && left.handlesPlaced && !left.points.empty()
                ? left.points.back().timeSeconds : seam-half;
            const auto to = right.independent && right.handlesPlaced && !right.points.empty()
                ? right.points.front().timeSeconds : seam+half;
            if (from > to) continue; // Invalid external/crossing data is not a reversed spline.
            auto& a = ensureMember(left); auto& b = ensureMember(right);
            const auto aLine = a.line, bLine = b.line;
            const auto low = aLine->midiAt(from), high = bLine->midiAt(to);
            auto joined = std::make_shared<SharedPitchLine>();
            for (const auto& piece : aLine->pieces)
                if (piece.from < from) joined->pieces.push_back(piece);
            PitchCurveEditPoint first {from, low}, last {to, high};
            first.shape = last.shape = PitchCurveShape::linear;
            joined->pieces.push_back({from, {first, last}});
            for (std::size_t p = 0; p < bLine->pieces.size(); ++p)
            {
                if (p+1 < bLine->pieces.size() && bLine->pieces[p+1].from <= to) continue;
                auto piece = bLine->pieces[p]; piece.from = std::max(to, piece.from);
                joined->pieces.push_back(std::move(piece));
            }
            for (auto& [id, member] : result.byNote)
                if (member.line == aLine || member.line == bLine)
                { member.line = joined; member.renderSharedCurve |= id == left.id || id == right.id; }
            a.joinsNext = true; b.joinsPrevious = true;
            a.drawTo = seam; b.drawFrom = seam;
            const auto boundaryHandle = [&](SharedPitchLineMember& member, const Entry& entry,
                                            double time, float midi, bool head)
            {
                if (!entry.independent || entry.handlesPlaced) return;
                if (member.nativeBoundaryAnchors.empty())
                {
                    member.nativeBoundaryAnchors = entry.points;
                    for (auto& p : member.nativeBoundaryAnchors) p.timeSeconds -= entry.start;
                }
                auto& anchors = member.nativeBoundaryAnchors;
                const auto local = time-entry.start;
                if (anchors.empty()) return;
                auto& point = head ? anchors.front() : anchors.back();
                point.timeSeconds = local; point.targetMidi = midi;
                member.nativeHandleTimes.clear();
                for (const auto& p : anchors) member.nativeHandleTimes.push_back(p.timeSeconds+entry.start);
            };
            boundaryHandle(a, left, from, low, false);
            boundaryHandle(b, right, to, high, true);
        }
    }
    return result;
}

SharedPitchLines sharedPitchLines(const TrackData& track, const juce::String& replacedNoteId,
                                  const std::vector<PitchCurveEditPoint>* replacedPoints, bool nativeEnvelopeGrouping)
{
    SharedPitchLines result;
    if (trackShowsAllNativeRegions(track))
        return nativeSharedPitchLines(track, replacedNoteId, replacedPoints, nativeEnvelopeGrouping);
    if (track.pitchAlgorithm != PitchAlgorithm::utau) return result;
    struct Entry
    {
        juce::String id;
        double start = 0.0;
        double end = 0.0;
        bool placed = false;
        bool rest = false;
        std::vector<PitchCurveEditPoint> points;   // absolute
    };
    std::vector<Entry> entries;
    for (const auto& clip : track.clips)
        for (const auto& note : clip.notes)
        {
            // A rest is sung by nobody, and nothing reaches across it.
            Entry entry;
            entry.rest = note.nativeUnpitched || backend::isRestLyric(note.label);
            entry.id = note.id;
            entry.start = clip.startSeconds + note.startSeconds;
            entry.end = entry.start + note.durationSeconds;
            const auto replaced = replacedPoints != nullptr && note.id == replacedNoteId
                && !replacedPoints->empty();
            entry.placed = replaced || !note.pitchControlPoints.empty();
            // In the order they are stored, not re-sorted: a UST can write a
            // bend whose widths run backwards, and its note has always been
            // sung by evaluating the points as they stand -- sorted, the same
            // points sing something else.  Its first point is the stored first,
            // as the renderer has always started a bend there.
            entry.points = replaced ? *replacedPoints : ownPitchPoints(note);
            for (auto& point : entry.points) point.timeSeconds += entry.start;
            if (!entry.points.empty()) entries.push_back(std::move(entry));
        }
    std::stable_sort(entries.begin(), entries.end(), [](const auto& left, const auto& right)
    {
        if (std::abs(left.start - right.start) > 1.0e-9) return left.start < right.start;
        return left.end < right.end;
    });

    // Joined where the two notes' own points overlap in time, whichever way
    // round: the next note's placed bend begins inside this one, or this
    // note's placed bend runs on past where the next one starts.  Either way
    // one line crosses the boundary, and one line is what both notes are then
    // sung along -- a bend drawn past a note's end used to be drawn and never
    // sung, because nothing carried it into the note it reached into.
    // A UST's rests come in as gaps, and a rest this short is an articulation
    // rather than a break in the line: real songs bend over rests of 46 and
    // 60 ms.  A tenth of a second or more is a real gap.
    constexpr double shortRestSeconds = 0.08;
    const auto joined = [&entries](std::size_t index)
    {
        const auto& left = entries[index];
        const auto& right = entries[index + 1];
        const auto gap = right.start - left.end;
        // Never borrow a line through a rest, a real gap or an independent
        // overlapping voice. Zero-length lead-in notes may share a beat.
        if (left.rest || right.rest || gap < -0.002 || gap > shortRestSeconds)
            return false;
        if (right.placed && right.points.front().timeSeconds <= left.end + 1.0e-9)
            return true;
        // Over a short rest only a bend drawn on both sides joins them: a
        // curve carries on into a note with no points of its own only where
        // the two really touch.
        if (gap > 0.002 && !(left.placed && right.placed)) return false;
        auto source = index;
        while (!entries[source].placed && source > 0)
        {
            const auto& previous = entries[source - 1];
            if (previous.rest || std::abs(entries[source].start - previous.end) > 0.002)
                return false;
            --source;
        }
        const auto& edited = entries[source];
        if (!edited.placed) return false;
        const auto last = std::max_element(edited.points.begin(), edited.points.end(),
            [](const auto& one, const auto& other)
            {
                return one.timeSeconds < other.timeSeconds;
            });
        return last != edited.points.end() && last->timeSeconds > right.start + 1.0e-9;
    };

    std::size_t first = 0;
    while (first < entries.size())
    {
        auto last = first;
        while (last + 1 < entries.size() && joined(last)) ++last;
        if (last == first)
        {
            ++first;
            continue;
        }
        const auto count = last - first + 1;
        // An unedited neighbour is a reader, not an overriding edit. Preserve
        // the preceding explicit curve through its last point, even when that
        // point lies inside (or beyond) the next note. Do not alter stored data.
        for (auto index = first + 1; index <= last; ++index)
        {
            auto& next = entries[index];
            auto source = index - 1;
            while (source > first && !entries[source].placed) --source;
            const auto& previous = entries[source];
            if (next.placed || !previous.placed) continue;
            const auto end = std::max_element(previous.points.begin(), previous.points.end(),
                [](const auto& a, const auto& b) { return a.timeSeconds < b.timeSeconds; });
            if (end == previous.points.end() || end->timeSeconds <= next.start) continue;
            const auto fallbackMidi = next.points.back().targetMidi;
            std::erase_if(next.points, [&](const auto& p) { return p.timeSeconds <= end->timeSeconds; });
            if (next.points.empty())
                next.points.push_back({ std::max(next.end, end->timeSeconds + 0.02),
                                       fallbackMidi });
        }
        // Who owns what: from its first point on, the latest note whose first
        // point has been reached.  A note whose every moment a later note has
        // already reached owns nothing.
        std::vector<double> firstPoint(count), ownFrom(count), ownUntil(count);
        for (std::size_t index = 0; index < count; ++index)
            firstPoint[index] = entries[first + index].points.front().timeSeconds;
        auto laterFirst = std::numeric_limits<double>::infinity();
        for (auto index = count; index-- > 0;)
        {
            ownFrom[index] = index == 0 ? -std::numeric_limits<double>::infinity()
                                        : firstPoint[index];
            ownUntil[index] = laterFirst;   // where the next owner takes over
            laterFirst = std::min(laterFirst, firstPoint[index]);
        }

        // A note with no point of its own inside its stretch owns nothing:
        // it would only be holding a point it no longer has there.
        std::vector<double> lastPoint(count, -std::numeric_limits<double>::infinity());
        for (std::size_t index = 0; index < count; ++index)
        {
            auto any = false;
            // Up to and including the moment the next note takes over: the
            // two may stand on one vertical line, and both count there.
            for (const auto& point : entries[first + index].points)
                if (point.timeSeconds >= ownFrom[index] - 1.0e-9
                    && point.timeSeconds <= ownUntil[index] + 1.0e-9)
                {
                    lastPoint[index] = any ? std::max(lastPoint[index], point.timeSeconds)
                                           : point.timeSeconds;
                    any = true;
                }
            if (!any) ownUntil[index] = ownFrom[index];
        }

        auto shared = std::make_shared<SharedPitchLine>();
        for (std::size_t index = 0; index < count; ++index)
        {
            if (!(ownUntil[index] > ownFrom[index])) continue;
            const auto& own = entries[first + index].points;
            SharedPitchLine::Piece piece;
            // The first stretch holds its start for as long as before it.
            piece.from = shared->pieces.empty() ? -std::numeric_limits<double>::infinity()
                                                : ownFrom[index];
            piece.points = own;
            shared->pieces.push_back(std::move(piece));
            if (!std::isfinite(ownUntil[index])) continue;
            // From this note's last point there to the first point of the note
            // that takes over, smoothly -- in the shape that first point gives
            // its incoming run.  Both ends are points on screen.
            std::size_t taker = index + 1;
            while (taker < count && std::abs(firstPoint[taker] - ownUntil[index]) > 1.0e-12)
                ++taker;
            if (taker >= count) continue;
            const auto& next = entries[first + taker].points;
            SharedPitchLine::Piece join;
            join.from = lastPoint[index];
            PitchCurveEditPoint from { lastPoint[index], evaluatePitchCurve(own, lastPoint[index]) };
            PitchCurveEditPoint into = next.front();
            into.timeSeconds = ownUntil[index];
            into.targetMidi = evaluatePitchCurve(next, ownUntil[index]);
            join.points = { from, into };
            if (ownUntil[index] - lastPoint[index] > 1.0e-9)
                shared->pieces.push_back(std::move(join));
        }
        for (std::size_t index = 0; index < count; ++index)
        {
            SharedPitchLineMember member;
            member.line = shared;
            const auto& own = entries[first + index].points;
            member.ownFrom = ownFrom[index];
            // Up to and including the moment the next note takes over.
            member.ownTo = std::isfinite(ownUntil[index]) ? ownUntil[index]
                                                          : std::numeric_limits<double>::infinity();
            if (!(ownUntil[index] > ownFrom[index]))
                member.ownTo = member.ownFrom;   // owns nothing
            member.drawFrom = std::max(ownFrom[index], own.front().timeSeconds);
            member.drawTo = std::isfinite(ownUntil[index]) ? ownUntil[index]
                                                           : own.back().timeSeconds;
            if (member.drawTo < member.drawFrom || !(ownUntil[index] > ownFrom[index]))
                member.drawTo = member.drawFrom;   // nothing of its own to draw
            member.joinsPrevious = index > 0;
            member.joinsNext = index + 1 < count;
            member.takeover = ownUntil[index];
            result.byNote[entries[first + index].id] = std::move(member);
        }
        first = last + 1;
    }
    return result;
}

float evaluatePitchCurve(const std::vector<PitchCurveEditPoint>& points,
                         double timeSeconds)
{
    if (points.empty()) return 60.0f;
    if (points.size() == 1) return points.front().targetMidi;
    const auto right = std::upper_bound(points.begin(), points.end(), timeSeconds,
        [](double value, const PitchCurveEditPoint& point)
        {
            return value < point.timeSeconds;
        });
    if (right == points.begin()) return points.front().targetMidi;
    if (right == points.end()) return points.back().targetMidi;
    const auto rightIndex = static_cast<std::size_t>(right - points.begin());
    const auto leftIndex = rightIndex - 1;
    const auto& left = points[leftIndex];
    const auto& next = points[rightIndex];
    const auto span = next.timeSeconds - left.timeSeconds;
    if (span <= 1.0e-9) return next.targetMidi;
    if(next.shape==PitchCurveShape::customBezier&&next.highPrecisionBezier)
    {
        const auto position=juce::jlimit(0.0,1.0,(timeSeconds-left.timeSeconds)/span);
        const auto x1=std::clamp<double>(next.bezierX1,0,1),x2=std::clamp<double>(next.bezierX2,0,1);
        const auto cubic=[](double t,double a,double b){const auto v=1-t;return 3*v*v*t*a+3*v*t*t*b+t*t*t;};
        double lo=0,hi=1;
        for(int i=0;i<56;++i){const auto t=(lo+hi)*.5;if(cubic(t,x1,x2)<position)lo=t;else hi=t;}
        const auto progress=cubic((lo+hi)*.5,next.bezierY1,next.bezierY2);
        return static_cast<float>(static_cast<double>(left.targetMidi)+(static_cast<double>(next.targetMidi)-left.targetMidi)*progress);
    }
    const auto u = static_cast<float>(juce::jlimit(0.0, 1.0,
        (timeSeconds - left.timeSeconds) / span));

    auto shaped = u;
    switch (next.shape)
    {
        case PitchCurveShape::linear:
        case PitchCurveShape::smooth:
        case PitchCurveShape::easeIn:
        case PitchCurveShape::easeOut:
        case PitchCurveShape::customBezier:
            shaped = shapedSegmentProgress(next.shape, u, next.bezierX1,
                                           next.bezierY1, next.bezierX2,
                                           next.bezierY2);
            break;
        case PitchCurveShape::natural:
        {
            // A monotone cubic Hermite curve.  Interior tangents use the
            // Fritsch-Carlson weighted harmonic mean; note endpoints settle
            // horizontally.  This keeps adjacent segments C1-smooth without
            // introducing accidental overshoot.  With only two points it is
            // exactly a smooth S transition.
            std::vector<double> intervals(points.size() - 1);
            std::vector<float> secants(points.size() - 1);
            for (std::size_t index = 0; index + 1 < points.size(); ++index)
            {
                intervals[index] = std::max(1.0e-9,
                    points[index + 1].timeSeconds - points[index].timeSeconds);
                secants[index] = (points[index + 1].targetMidi
                    - points[index].targetMidi) / static_cast<float>(intervals[index]);
            }
            std::vector<float> tangents(points.size(), 0.0f);
            for (std::size_t index = 1; index + 1 < points.size(); ++index)
            {
                const auto before = secants[index - 1];
                const auto after = secants[index];
                if (before == 0.0f || after == 0.0f
                    || std::signbit(before) != std::signbit(after))
                    continue;
                const auto firstWeight = 2.0 * intervals[index] + intervals[index - 1];
                const auto secondWeight = intervals[index] + 2.0 * intervals[index - 1];
                tangents[index] = static_cast<float>((firstWeight + secondWeight)
                    / (firstWeight / before + secondWeight / after));
            }
            const auto h00 = 2.0f * u * u * u - 3.0f * u * u + 1.0f;
            const auto h10 = u * u * u - 2.0f * u * u + u;
            const auto h01 = -2.0f * u * u * u + 3.0f * u * u;
            const auto h11 = u * u * u - u * u;
            return h00 * left.targetMidi
                + h10 * static_cast<float>(span) * tangents[leftIndex]
                + h01 * next.targetMidi
                + h11 * static_cast<float>(span) * tangents[rightIndex];
        }
    }
    return left.targetMidi + (next.targetMidi - left.targetMidi) * shaped;
}

juce::String pitchCurveShapeName(PitchCurveShape value)
{
    switch (value)
    {
        case PitchCurveShape::natural: return "natural";
        case PitchCurveShape::linear: return "linear";
        case PitchCurveShape::smooth: return "smooth";
        case PitchCurveShape::easeIn: return "ease-in";
        case PitchCurveShape::easeOut: return "ease-out";
        case PitchCurveShape::customBezier: return "custom-bezier";
    }
    return "natural";
}

PitchCurveShape parsePitchCurveShape(const juce::String& value)
{
    if (value == "linear") return PitchCurveShape::linear;
    if (value == "smooth") return PitchCurveShape::smooth;
    if (value == "ease-in") return PitchCurveShape::easeIn;
    if (value == "ease-out") return PitchCurveShape::easeOut;
    if (value == "custom-bezier") return PitchCurveShape::customBezier;
    return PitchCurveShape::natural;
}

// The two above are named in the project file and over MCP, so they belong to
// the interface; everything below is this file's own business.
namespace
{
juce::String pitchAlgorithmName(PitchAlgorithm value)
{
    switch (value)
    {
        case PitchAlgorithm::mld5: return "mld5";
        case PitchAlgorithm::mld3: return "mld3";
        case PitchAlgorithm::nsfHifigan: return "nsf-hifigan";
        case PitchAlgorithm::world: return "world";
        case PitchAlgorithm::vocalShifter: return "vslib";
        case PitchAlgorithm::llsm2: return "llsm2";
        case PitchAlgorithm::utau: return "utau";
    }
    return "llsm2";
}

PitchAlgorithm parsePitchAlgorithm(const juce::String& value)
{
    if (value == "nsf-hifigan") return PitchAlgorithm::nsfHifigan;
    if (value == "mld3") return PitchAlgorithm::mld3;
    if (value == "mld5") return PitchAlgorithm::mld5;
    if (value == "world") return PitchAlgorithm::world;
    if (value == "vslib") return PitchAlgorithm::vocalShifter;
    if (value == "llsm2") return PitchAlgorithm::llsm2;
    if (value == "utau") return PitchAlgorithm::utau;
    return PitchAlgorithm::llsm2;
}

juce::String stretchAlgorithmName(StretchAlgorithm value)
{
    switch (value)
    {
        case StretchAlgorithm::melodyneHybrid: return "melodyne-hybrid";
        case StretchAlgorithm::variableMelHop: return "variable-mel-hop";
        case StretchAlgorithm::loop: return "loop";
        case StretchAlgorithm::soundTouch: return "soundtouch";
        case StretchAlgorithm::nsfShiftThenSplice: return "nsf-shift-then-splice";
        case StretchAlgorithm::hifiShifterMel: return "hifishifter-mel";
        case StretchAlgorithm::wcsNonLoop: return "wcsndm-nonloop";
    }
    return "melodyne-hybrid";
}

StretchAlgorithm parseStretchAlgorithm(const juce::String& value)
{
    if (value == "variable-mel-hop") return StretchAlgorithm::variableMelHop;
    if (value == "loop") return StretchAlgorithm::loop;
    if (value == "soundtouch") return StretchAlgorithm::soundTouch;
    if (value == "nsf-shift-then-splice") return StretchAlgorithm::nsfShiftThenSplice;
    if (value == "hifishifter-mel") return StretchAlgorithm::hifiShifterMel;
    if (value == "wcsndm-nonloop") return StretchAlgorithm::wcsNonLoop;
    return StretchAlgorithm::melodyneHybrid;
}

juce::String renderOrderName(RenderOrder value)
{
    switch (value)
    {
        case RenderOrder::processThenSplice: return "process-then-splice";
        case RenderOrder::stretchSpliceThenPitch: return "stretch-splice-then-pitch";
    }
    return "process-then-splice";
}

RenderOrder parseRenderOrder(const juce::String& value)
{
    if (value == "stretch-splice-then-pitch") return RenderOrder::stretchSpliceThenPitch;
    return RenderOrder::processThenSplice;
}
}

namespace
{
struct GraphPitchEndpoint
{
    const NoteData* note=nullptr;const ClipData* parent=nullptr;
    double start=0,end=0,anchor=0;
    bool sourceOwnerValid=true;
};
std::vector<GraphPitchEndpoint> graphPitchEndpoints(const TrackData& track)
{
    std::vector<GraphPitchEndpoint> result;
    for(const auto& parent:track.clips)for(const auto& n:parent.notes)
    {
        const auto start=parent.startSeconds+n.startSeconds;double anchor=start;
        const ClipData* source=&parent;
        if(!parent.parts.empty())
        {
            source=nullptr;for(const auto& part:parent.parts)if(part.id==n.clipPartId){if(source){source=nullptr;break;}source=&part;}
        }
        if(source&&trackShowsAllNativeRegions(track)&&source->sourceFile!=juce::File{}
            &&!source->sourceFile.hasFileExtension("ust;mid;midi;mpd;hjpx;hspx"))
            anchor=parent.startSeconds+nativeNotePreutteranceTarget(n,parent);
        result.push_back({&n,&parent,start,start+n.durationSeconds,anchor,!trackShowsAllNativeRegions(track)||source!=nullptr});
    }
    std::stable_sort(result.begin(),result.end(),[](const auto& a,const auto& b){return a.start<b.start;});return result;
}
juce::String graphPitchPairIssue(const TrackData& track,const std::vector<GraphPitchEndpoint>& all,std::size_t a,std::size_t b)
{
    if(!track.compose||track.accompaniment)return "The track has no note-pitch renderer";
    if(b!=a+1)return "Portamento requires consecutive notes, with no intervening note or rest";
    const auto& left=all[a];const auto& right=all[b];
    if(!left.sourceOwnerValid||!right.sourceOwnerValid)return "Ambiguous physical source ownership for the P anchor";
    if(!std::isfinite(left.start)||!std::isfinite(left.end)||!std::isfinite(right.start)||!std::isfinite(right.end)||!std::isfinite(right.anchor)
        ||left.end-left.start<.001||right.end-right.start<.001)return "Invalid or zero-duration pitch endpoint";
    if(left.end>right.start+1.e-7)return "Overlapping notes cannot share a forced portamento";
    if(right.start-left.end>.001)return "A gap is not filled by forcing pitch; move the notes together explicitly";
    const auto silent=[&](const NoteData& n){return n.nativeUnpitched||n.pronunciationKind==PronunciationKind::breath
        ||(trackUsesVoicebankSynthesis(track)&&backend::isRestLyric(n.label));};
    if(silent(*left.note)||silent(*right.note))return "A rest, breath or unpitched source cannot be forced voiced";
    return {};
}
float graphBaseMidiAt(const SharedPitchLines& lines,const GraphPitchEndpoint& note,double t,
                      const juce::String& replacedId,const std::vector<PitchCurveEditPoint>* replaced)
{
    if(const auto* member=lines.memberFor(note.note->id))return member->line->midiAt(t);
    return evaluatePitchCurve(replaced&&!replaced->empty()&&note.note->id==replacedId?*replaced:ownPitchPoints(*note.note),t-note.start);
}
bool finiteGraphPoint(const PitchCurveEditPoint& p)
{
    return std::isfinite(p.timeSeconds)&&std::isfinite(p.targetMidi)&&std::isfinite(p.bezierX1)&&std::isfinite(p.bezierX2)
        &&std::isfinite(p.bezierY1)&&std::isfinite(p.bezierY2)&&p.bezierX1>=0&&p.bezierX1<=1&&p.bezierX2>=0&&p.bezierX2<=1
        &&static_cast<int>(p.shape)>=0&&static_cast<int>(p.shape)<=static_cast<int>(PitchCurveShape::customBezier);
}
bool sameGraphPitchData(const NativeConnection& a,const NativeConnection& b)
{
    if(a.id!=b.id||a.pitchCurveEndpointRelative!=b.pitchCurveEndpointRelative||a.pitchStartOffsetSeconds!=b.pitchStartOffsetSeconds
        ||a.pitchEndOffsetSeconds!=b.pitchEndOffsetSeconds||a.pitchCurve.size()!=b.pitchCurve.size())return false;
    for(std::size_t i=0;i<a.pitchCurve.size();++i)
    {
        const auto& x=a.pitchCurve[i];const auto& y=b.pitchCurve[i];
        if(x.timeSeconds!=y.timeSeconds||x.targetMidi!=y.targetMidi||x.shape!=y.shape||x.bezierX1!=y.bezierX1||x.bezierX2!=y.bezierX2
            ||x.bezierY1!=y.bezierY1||x.bezierY2!=y.bezierY2||x.diffSingerRestoreSupport!=y.diffSingerRestoreSupport||x.highPrecisionBezier!=y.highPrecisionBezier)return false;
    }
    return true;
}
}

std::vector<ResolvedConnectionPitchCurve> resolvedConnectionPitchCurves(const TrackData& track,
    const std::vector<NativeConnection>& connections,const juce::String& replacedId,const std::vector<PitchCurveEditPoint>* replaced)
{
    const auto endpoints=graphPitchEndpoints(track);const auto base=sharedPitchLines(track,replacedId,replaced,false);
    std::vector<ResolvedConnectionPitchCurve> result;
    for(const auto& connection:connections)
    {
        if(connection.pitchCurve.empty())continue;
        std::vector<std::size_t> left,right;
        for(std::size_t i=0;i<endpoints.size();++i){if(endpoints[i].note->id==connection.leftNoteId)left.push_back(i);if(endpoints[i].note->id==connection.rightNoteId)right.push_back(i);}
        if(left.empty()&&right.empty())continue;
        ResolvedConnectionPitchCurve resolved;resolved.connectionId=connection.id;resolved.leftNoteId=connection.leftNoteId;resolved.rightNoteId=connection.rightNoteId;
        if(left.size()!=1||right.size()!=1){resolved.issue="Missing, ambiguous or cross-track pitch endpoint";result.push_back(std::move(resolved));continue;}
        const auto& a=endpoints[left.front()];const auto& b=endpoints[right.front()];resolved.anchorSeconds=b.anchor;
        resolved.issue=graphPitchPairIssue(track,endpoints,left.front(),right.front());
        if(resolved.issue.isNotEmpty()){result.push_back(std::move(resolved));continue;}
        if(connection.pitchCurve.size()<2||connection.pitchCurve.size()>2048){resolved.issue="A graph pitch curve needs 2..2048 controls";result.push_back(std::move(resolved));continue;}
        bool valid=true;double previous=-std::numeric_limits<double>::infinity();
        for(const auto& p:connection.pitchCurve){valid&=finiteGraphPoint(p)&&p.timeSeconds>previous;previous=p.timeSeconds;}
        if(!valid){resolved.issue="Invalid graph pitch controls";result.push_back(std::move(resolved));continue;}
        if(connection.pitchCurveEndpointRelative)
        {
            const auto& points=connection.pitchCurve;
            if(!std::isfinite(connection.pitchStartOffsetSeconds)||!std::isfinite(connection.pitchEndOffsetSeconds)
                ||std::abs(points.front().timeSeconds)>1.e-8||std::abs(points.back().timeSeconds-1)>1.e-8
                ||std::abs(points.front().targetMidi)>1.e-4f||std::abs(points.back().targetMidi)>1.e-4f)
                resolved.issue="Invalid live-endpoint graph encoding";
            else
            {
                resolved.fromSeconds=juce::jlimit(a.start,a.end,a.end+connection.pitchStartOffsetSeconds);
                resolved.toSeconds=juce::jlimit(b.start,b.end,b.anchor+connection.pitchEndOffsetSeconds);
                const auto low=graphBaseMidiAt(base,a,resolved.fromSeconds,replacedId,replaced);
                const auto high=graphBaseMidiAt(base,b,resolved.toSeconds,replacedId,replaced);
                for(auto p:points)
                {
                    const auto u=p.timeSeconds;p.timeSeconds=resolved.fromSeconds+u*(resolved.toSeconds-resolved.fromSeconds);
                    p.targetMidi=static_cast<float>(low+(high-low)*u)+p.targetMidi;resolved.points.push_back(p);
                }
            }
        }
        else
        {
            resolved.points=connection.pitchCurve;resolved.fromSeconds=resolved.points.front().timeSeconds;resolved.toSeconds=resolved.points.back().timeSeconds;
            if(resolved.fromSeconds<a.start-1.e-8||resolved.fromSeconds>a.end+1.e-8||resolved.toSeconds<b.start-1.e-8||resolved.toSeconds>b.end+1.e-8)
                resolved.issue="The legacy absolute bridge no longer fits its live endpoints";
        }
        if(resolved.toSeconds-resolved.fromSeconds<.001)resolved.issue="The bridge has less than 1ms of target time";
        for(std::size_t i=0;i<endpoints.size();++i)if(i!=left.front()&&i!=right.front()
            &&endpoints[i].start<resolved.toSeconds-1.e-8&&endpoints[i].end>resolved.fromSeconds+1.e-8)
            resolved.issue="An overlapping third voice makes this bridge window ambiguous";
        resolved.active=resolved.issue.isEmpty();result.push_back(std::move(resolved));
    }
    std::vector<bool> overlaps(result.size(),false);
    for(std::size_t i=0;i<result.size();++i)for(std::size_t j=0;j<i;++j)
        if(result[i].active&&result[j].active&&(result[i].leftNoteId==result[j].leftNoteId||result[i].leftNoteId==result[j].rightNoteId
            ||result[i].rightNoteId==result[j].leftNoteId||result[i].rightNoteId==result[j].rightNoteId)
            &&std::min(result[i].toSeconds,result[j].toSeconds)>std::max(result[i].fromSeconds,result[j].fromSeconds)+1.e-8)
        {overlaps[i]=overlaps[j]=true;}
    for(std::size_t i=0;i<result.size();++i)if(overlaps[i]){result[i].active=false;result[i].issue="Incident graph bridge windows overlap; shorten a bridge explicitly";}
    return result;
}

SharedPitchLines sharedPitchLines(const TrackData& track,const std::vector<NativeConnection>& connections,
    const juce::String& replacedId,const std::vector<PitchCurveEditPoint>* replaced)
{
    auto result=sharedPitchLines(track,replacedId,replaced,false);const auto endpoints=graphPitchEndpoints(track);
    auto curves=resolvedConnectionPitchCurves(track,connections,replacedId,replaced);
    std::stable_sort(curves.begin(),curves.end(),[](const auto& a,const auto& b){return a.fromSeconds<b.fromSeconds;});
    for(const auto& curve:curves)if(curve.active)for(const auto& id:{curve.leftNoteId,curve.rightNoteId})
    {
        auto found=result.byNote.find(id);
        if(found==result.byNote.end())
        {
            const auto endpoint=std::find_if(endpoints.begin(),endpoints.end(),[&](const auto& p){return p.note->id==id;});
            if(endpoint==endpoints.end())continue;
            auto own=replaced&&!replaced->empty()&&id==replacedId?*replaced:ownPitchPoints(*endpoint->note);for(auto& p:own)p.timeSeconds+=endpoint->start;
            auto line=std::make_shared<SharedPitchLine>();line->pieces.push_back({-std::numeric_limits<double>::infinity(),std::move(own)});
            SharedPitchLineMember member;member.line=line;member.drawFrom=endpoint->start;member.drawTo=endpoint->end;
            for(const auto& p:line->pieces.front().points)member.nativeHandleTimes.push_back(p.timeSeconds);
            found=result.byNote.emplace(id,std::move(member)).first;
        }
        const auto old=found->second.line;if(!old||old->pieces.empty())continue;
        auto line=std::make_shared<SharedPitchLine>();std::size_t restore=0;
        for(std::size_t i=0;i<old->pieces.size();++i)
        {
            if(old->pieces[i].from<curve.fromSeconds)line->pieces.push_back(old->pieces[i]);
            if(old->pieces[i].from<=curve.toSeconds)restore=i;
        }
        line->pieces.push_back({curve.fromSeconds,curve.points});
        auto restored=old->pieces[restore];restored.from=curve.toSeconds;line->pieces.push_back(std::move(restored));
        for(const auto& piece:old->pieces)if(piece.from>curve.toSeconds)line->pieces.push_back(piece);
        if(found->second.connectionPitchWindows.empty())found->second.connectionPitchOnly=!found->second.renderSharedCurve;
        found->second.line=std::move(line);found->second.renderSharedCurve=true;
        found->second.nativeSharedCurve=trackShowsAllNativeRegions(track);
        found->second.connectionPitchWindows.emplace_back(curve.fromSeconds,curve.toSeconds);
        // Graph handles remain owned/editable on the graph, never impersonated
        // as a second set of writable note-local pitch controls.
    }
    return result;
}

juce::String connectionPitchDigest(const TrackData& track,const std::vector<NativeConnection>& connections,const ClipData& clip)
{
    juce::MemoryOutputStream stream;bool any=false;
    for(const auto& curve:resolvedConnectionPitchCurves(track,connections))
    {
        if(std::none_of(clip.notes.begin(),clip.notes.end(),[&](const auto& n){return n.id==curve.leftNoteId||n.id==curve.rightNoteId;}))continue;
        any=true;stream.writeString(curve.connectionId);stream.writeString(curve.leftNoteId);stream.writeString(curve.rightNoteId);
        stream.writeBool(curve.active);stream.writeString(curve.issue);stream.writeDouble(curve.fromSeconds);stream.writeDouble(curve.toSeconds);stream.writeDouble(curve.anchorSeconds);
        for(const auto& p:curve.points){stream.writeDouble(p.timeSeconds);stream.writeFloat(p.targetMidi);stream.writeInt(static_cast<int>(p.shape));
            stream.writeFloat(p.bezierX1);stream.writeFloat(p.bezierY1);stream.writeFloat(p.bezierX2);stream.writeFloat(p.bezierY2);
            if(p.highPrecisionBezier)stream.writeString("high-precision-bezier-v1");}
    }
    return any?juce::SHA256(stream.getData(),stream.getDataSize()).toHexString():juce::String{};
}

namespace
{
bool prepareConnectionPitchEdit(const TrackData& track,std::vector<NativeConnection>& graph,const juce::String& id,
                                 const std::vector<PitchCurveEditPoint>& absolute,juce::String& error)
{
    error.clear();const auto found=std::find_if(graph.begin(),graph.end(),[&](const auto& c){return c.id==id;});
    if(found==graph.end()||std::count_if(graph.begin(),graph.end(),[&](const auto& c){return c.id==id;})!=1){error="Unknown or ambiguous connection";return false;}
    const auto endpoints=graphPitchEndpoints(track);std::size_t a=endpoints.size(),b=endpoints.size();int countA=0,countB=0;
    for(std::size_t i=0;i<endpoints.size();++i){if(endpoints[i].note->id==found->leftNoteId){a=i;++countA;}if(endpoints[i].note->id==found->rightNoteId){b=i;++countB;}}
    if(countA!=1||countB!=1){error="Missing or ambiguous connection endpoints";return false;}
    error=graphPitchPairIssue(track,endpoints,a,b);if(error.isNotEmpty())return false;
    if(absolute.size()<2||absolute.size()>2048){error="A bridge needs 2..2048 controls";return false;}
    double previous=-std::numeric_limits<double>::infinity();
    for(const auto& p:absolute)if(!finiteGraphPoint(p)||p.timeSeconds<=previous||p.targetMidi<0||p.targetMidi>127)
    {error="Invalid editable bridge points";return false;}else previous=p.timeSeconds;
    const auto prior=resolvedConnectionPitchCurves(track,graph);
    const auto current=std::find_if(prior.begin(),prior.end(),[&](const auto& c){return c.connectionId==id;});
    if(current!=prior.end()&&current->active&&current->points.size()==absolute.size())
    {
        bool same=true;
        for(std::size_t i=0;i<absolute.size();++i)
        {
            const auto& x=current->points[i];const auto& y=absolute[i];
            same&=std::abs(x.timeSeconds-y.timeSeconds)<1.e-10&&x.targetMidi==y.targetMidi&&x.shape==y.shape
                &&x.bezierX1==y.bezierX1&&x.bezierY1==y.bezierY1&&x.bezierX2==y.bezierX2&&x.bezierY2==y.bezierY2&&x.highPrecisionBezier==y.highPrecisionBezier;
        }
        if(same)return false;
    }
    const auto from=absolute.front().timeSeconds,to=absolute.back().timeSeconds;
    if(from<endpoints[a].start-1.e-8||from>endpoints[a].end+1.e-8||to<endpoints[b].start-1.e-8||to>endpoints[b].end+1.e-8||to-from<.001)
    {error="Bridge endpoints must remain inside their own note bodies with at least 1ms duration";return false;}
    const auto base=sharedPitchLines(track);const auto low=graphBaseMidiAt(base,endpoints[a],from,{},nullptr),high=graphBaseMidiAt(base,endpoints[b],to,{},nullptr);
    if(std::abs(absolute.front().targetMidi-low)>.0002f||std::abs(absolute.back().targetMidi-high)>.0002f)
    {error="Outer endpoint pitch follows the live score; edit endpoint time/curvature or interior pitch instead";return false;}
    auto candidate=*found;candidate.pitchCurve.clear();candidate.pitchCurveEndpointRelative=true;
    candidate.pitchStartOffsetSeconds=from-endpoints[a].end;candidate.pitchEndOffsetSeconds=to-endpoints[b].anchor;
    for(auto p:absolute)
    {
        const auto u=(p.timeSeconds-from)/(to-from);p.timeSeconds=u;p.targetMidi-=static_cast<float>(low+(high-low)*u);candidate.pitchCurve.push_back(p);
    }
    candidate.pitchCurve.front().timeSeconds=0;candidate.pitchCurve.back().timeSeconds=1;
    candidate.pitchCurve.front().targetMidi=candidate.pitchCurve.back().targetMidi=0;
    const auto old=*found;*found=candidate;
    const auto resolved=resolvedConnectionPitchCurves(track,graph);const auto check=std::find_if(resolved.begin(),resolved.end(),[&](const auto& c){return c.connectionId==id;});
    if(check==resolved.end()||!check->active){*found=old;error=check==resolved.end()?juce::String("Bridge cannot be resolved"):check->issue;return false;}
    return !sameGraphPitchData(old,candidate);
}
const TrackData* graphConnectionTrack(const ProjectData& project,const std::vector<NativeConnection>& graph,const juce::String& id,juce::String& error)
{
    const auto c=std::find_if(graph.begin(),graph.end(),[&](const auto& c){return c.id==id;});
    if(c==graph.end()){error="Unknown connection";return nullptr;}
    const TrackData* owner=nullptr;
    for(const auto& t:project.tracks)
    {
        int left=0,right=0;for(const auto& p:t.clips)for(const auto& n:p.notes){left+=n.id==c->leftNoteId;right+=n.id==c->rightNoteId;}
        if(left&&right){if(owner||left!=1||right!=1){error="Ambiguous connection track";return nullptr;}owner=&t;}
    }
    if(!owner)error="Connection endpoints are not on one track";return owner;
}
}

double ProjectData::durationSeconds() const
{
    double duration = 8.0;
    for (const auto& track : tracks)
        for (const auto& clip : track.clips)
            duration = std::max(duration, clip.startSeconds + clip.durationSeconds);
    return duration;
}

double ProjectData::secondsForQuarterPosition(double quarterPosition) const
{
    const auto baseTempo = juce::jlimit(20.0, 400.0, bpm);
    if (quarterPosition <= 0.0)
        return beatOriginSeconds + quarterPosition * 60.0 / baseTempo;

    auto seconds = beatOriginSeconds;
    auto previousQuarter = 0.0;
    auto tempo = baseTempo;
    for (const auto& change : tempoChanges)
    {
        const auto changeQuarter = std::max(0.0, change.quarterPosition);
        if (changeQuarter <= previousQuarter + 1.0e-9)
        {
            tempo = juce::jlimit(20.0, 400.0, change.bpm);
            continue;
        }
        if (quarterPosition <= changeQuarter)
            return seconds + (quarterPosition - previousQuarter) * 60.0 / tempo;
        seconds += (changeQuarter - previousQuarter) * 60.0 / tempo;
        previousQuarter = changeQuarter;
        tempo = juce::jlimit(20.0, 400.0, change.bpm);
    }
    return seconds + (quarterPosition - previousQuarter) * 60.0 / tempo;
}

double ProjectData::quarterPositionForSeconds(double targetSeconds) const
{
    const auto baseTempo = juce::jlimit(20.0, 400.0, bpm);
    if (targetSeconds <= beatOriginSeconds)
        return (targetSeconds - beatOriginSeconds) * baseTempo / 60.0;

    auto seconds = beatOriginSeconds;
    auto previousQuarter = 0.0;
    auto tempo = baseTempo;
    for (const auto& change : tempoChanges)
    {
        const auto changeQuarter = std::max(0.0, change.quarterPosition);
        if (changeQuarter <= previousQuarter + 1.0e-9)
        {
            tempo = juce::jlimit(20.0, 400.0, change.bpm);
            continue;
        }
        const auto changeSeconds = seconds
            + (changeQuarter - previousQuarter) * 60.0 / tempo;
        if (targetSeconds <= changeSeconds)
            return previousQuarter + (targetSeconds - seconds) * tempo / 60.0;
        seconds = changeSeconds;
        previousQuarter = changeQuarter;
        tempo = juce::jlimit(20.0, 400.0, change.bpm);
    }
    return previousQuarter + (targetSeconds - seconds) * tempo / 60.0;
}

double ProjectData::tempoAtQuarterPosition(double quarterPosition) const
{
    auto tempo = juce::jlimit(20.0, 400.0, bpm);
    for (const auto& change : tempoChanges)
    {
        if (change.quarterPosition > quarterPosition + 1.0e-9) break;
        tempo = juce::jlimit(20.0, 400.0, change.bpm);
    }
    return tempo;
}

double ProjectData::tempoAtSeconds(double seconds) const
{
    return tempoAtQuarterPosition(quarterPositionForSeconds(seconds));
}

ProjectModel::ProjectModel()
{
    project.name = "Untitled";
}

ProjectData ProjectModel::snapshot() const
{
    const juce::ScopedLock guard(lock);
    return project;
}

std::uint64_t ProjectModel::revisionNumber() const
{
    const juce::ScopedLock guard(lock);
    return revision;
}

void ProjectModel::pushUndoLocked()
{
    if (suppressNestedUndo) return;
    undoHistory.push_back(project);
    if (undoHistory.size() > maxHistory)
        undoHistory.erase(undoHistory.begin());
    redoHistory.clear();
    ++revision;
}

void ProjectModel::replace(ProjectData replacement)
{
    replaceInternal(std::move(replacement), false);
}

void ProjectModel::resetDocument(ProjectData replacement)
{
    replaceInternal(std::move(replacement), true);
}

void ProjectModel::replaceInternal(ProjectData replacement, bool newDocument)
{
    for (auto& track : replacement.tracks)
    {
        if (track.accompaniment) { track.compose = false; track.referenceOnly = false; }
        if(!track.trackType)
        {
            if(!track.accompaniment&&track.voicebankDirectory.getChildFile("dsconfig.yaml").existsAsFile())track.utauMode=UtauMode::mou;
            track.trackType=effectiveTrackType(track);
        }
    }
    {
        const juce::ScopedLock guard(lock);
        if (newDocument) { undoHistory.clear(); redoHistory.clear(); ++revision; }
        else
        {
            replacement.assetManagerVisible = project.assetManagerVisible;
            pushUndoLocked();
        }
        project = std::move(replacement);
    }
    sendChangeMessage();
}

bool ProjectModel::setHamoodState(const juce::String& state, juce::String& error)
{
    const auto parsed=juce::JSON::parse(state);
    if(!hamoodstate::validate(parsed,error))return false;
    const auto canonical=hamoodstate::encode(parsed);
    {const juce::ScopedLock guard(lock);if(project.hamoodState==canonical)return true;pushUndoLocked();project.hamoodState=canonical;}
    sendChangeMessage();return true;
}

void ProjectModel::clear()
{
    resetDocument();
}

void ProjectModel::setAssetManagerVisible(bool visible)
{
    const juce::ScopedLock guard(lock);
    if (project.assetManagerVisible == visible) return;
    project.assetManagerVisible = visible;
    ++revision;
}

bool ProjectModel::undo()
{
    {
        const juce::ScopedLock guard(lock);
        if (undoHistory.empty()) return false;
        redoHistory.push_back(project);
        if (redoHistory.size() > maxHistory) redoHistory.erase(redoHistory.begin());
        const auto visible = project.assetManagerVisible;
        project = std::move(undoHistory.back());
        project.assetManagerVisible = visible;
        undoHistory.pop_back();
        ++revision;
    }
    sendChangeMessage();
    return true;
}

bool ProjectModel::redo()
{
    {
        const juce::ScopedLock guard(lock);
        if (redoHistory.empty()) return false;
        undoHistory.push_back(project);
        if (undoHistory.size() > maxHistory) undoHistory.erase(undoHistory.begin());
        const auto visible = project.assetManagerVisible;
        project = std::move(redoHistory.back());
        project.assetManagerVisible = visible;
        redoHistory.pop_back();
        ++revision;
    }
    sendChangeMessage();
    return true;
}

bool ProjectModel::canUndo() const
{
    const juce::ScopedLock guard(lock);
    return !undoHistory.empty();
}

bool ProjectModel::canRedo() const
{
    const juce::ScopedLock guard(lock);
    return !redoHistory.empty();
}

juce::String ProjectModel::makeId(const char* prefix)
{
    return juce::String(prefix) + "_" + juce::Uuid().toString().removeCharacters("-");
}

namespace
{
struct MaterialAliasProjection
{
    juce::String bound, label, previousEffective;
    bool missing = false;
    bool labelFollowsAlias = false;
};
MaterialAliasProjection materialAliasProjection(const NoteData& note,const SampleRegionSetting& row)
{
    const auto primary=row.name.trim().isEmpty()?juce::String("-"):row.name.trim();
    const auto effective=note.materialBoundAlias.isNotEmpty()?note.materialBoundAlias
        :note.materialLabelFollowsAlias?note.label:primary;
    const auto bound=note.materialAliasFollowsPrimary?primary:effective;
    // Carried Melodyne placements were registered with a '-' display snapshot.
    // Naming that placeholder starts following the primary alias; explicitly
    // renamed note labels remain independent of subsequent source publications.
    const bool followsLabel=note.materialLabelFollowsAlias
        || (note.materialAliasFollowsPrimary&&note.label=="-"&&effective=="-"&&primary!="-");
    return {bound,followsLabel?bound:note.label,effective,
        !note.materialAliasFollowsPrimary&&bound!=primary&&!row.additionalAliases.contains(bound),followsLabel};
}
// Source-snapshot whitelist shared by source publication and note-local edits.
// This deliberately does not initialize/reset any musical target or source F0.
void applyMaterialAnnotationSnapshot(NoteData& note,const SampleRegionSetting& row,
    bool intact,const std::vector<NativeSegment>& segments)
{
    const auto alias=materialAliasProjection(note,row);
    note.label=alias.label;note.materialBoundAlias=alias.bound;note.materialLabelFollowsAlias=alias.labelFollowsAlias;note.nativeRole=row.role;
    note.nativeProvenance=row.provenance;note.nativeConfidence=row.confidence;
    note.materialAudioId=row.materialAudioId;note.materialRegionId=row.materialRegionId;
    note.materialAdditionalAliases=row.additionalAliases;note.materialConnectedNextRegionId=row.connectedNextRegionId;
    note.pronunciationKind=row.pronunciationKind;note.simplifiedConsonant=row.simplifiedConsonant;note.vowelSplitsExplicit=row.vowelSplitsExplicit;
    note.materialReferenceUnresolved=false;note.materialSourceStartSeconds=row.regionStartSeconds;note.materialSourceEndSeconds=row.regionEndSeconds;
    if(intact){note.nativeSourceStartSeconds=row.regionStartSeconds;note.nativeSourceEndSeconds=row.regionEndSeconds;note.nativeSegments=segments;}
}

// Preserve hidden absolute-file trim anchors while publishing a validated clock.
// Both full-source updates and local edits use this one ownership operation.
bool publishMaterialSourceClock(ClipData& owner,const std::vector<SourceTimePoint>& oldClock,
    std::vector<SourceTimePoint> newClock,double first,double last,juce::String& error)
{
    constexpr double epsilon=1.e-8;
    if(!owner.nativeTrimClock.empty())
    {
        std::vector<SourceTimePoint> trim;
        for(const auto& point:owner.nativeTrimClock)
            if(point.targetSeconds<oldClock.front().targetSeconds-epsilon||point.targetSeconds>oldClock.back().targetSeconds+epsilon)trim.push_back(point);
        for(auto point:newClock){point.sourceSeconds+=first;trim.push_back(point);}
        std::stable_sort(trim.begin(),trim.end(),[](const auto& a,const auto& b){return a.targetSeconds<b.targetSeconds;});
        for(std::size_t i=1;i<trim.size();++i)if(trim[i].sourceSeconds<=trim[i-1].sourceSeconds+epsilon)
        {error="Boundary conflicts with retained trim history.";return false;}
        owner.nativeTrimClock=std::move(trim);
    }
    owner.sourceTimeMap=std::move(newClock);owner.sourceOffsetSeconds=first;owner.sourceDurationSeconds=last-first;return true;
}

// Pure annotation-to-native-note builder shared by ordinary annotated audio
// and explicit pronunciation insertion. The caller owns identity and timing.
void initialiseMaterialNote(NoteData& note, const SampleRegionSetting& row,
                            bool legacyUnpitchedMeasured = false)
{
    note.label = row.name.trim().isEmpty() ? "-" : row.name.trim();
    note.materialBoundAlias=note.label;note.materialAliasFollowsPrimary=true;note.materialLabelFollowsAlias=true;
    note.nativeRole = row.role; note.nativeUnpitched = row.nativeUnpitched;
    note.nativeProvenance = row.provenance; note.nativeConfidence = row.confidence;
    note.materialAudioId = row.materialAudioId; note.materialRegionId = row.materialRegionId;
    note.materialSourceStartSeconds = note.nativeSourceStartSeconds = row.regionStartSeconds;
    note.materialSourceEndSeconds = note.nativeSourceEndSeconds = row.regionEndSeconds;
    note.materialAdditionalAliases = row.additionalAliases;
    note.materialConnectedNextRegionId = row.connectedNextRegionId;
    note.pronunciationKind=row.pronunciationKind;note.simplifiedConsonant=row.simplifiedConsonant;note.vowelSplitsExplicit=row.vowelSplitsExplicit;
    note.consonantSeconds = juce::jlimit(0.0, note.durationSeconds, row.fixedDurationSeconds);
    const auto storedSource = row.melodyneOriginalPitchCenterCents > 0.0
        ? static_cast<float>(row.melodyneOriginalPitchCenterCents / 100.0)
        : row.melodynePitchCenterCents > 0.0
            ? static_cast<float>(row.melodynePitchCenterCents / 100.0) : 60.0f;
    note.sourceMidiCenter = row.melodyneOriginalPitchCenterCents > 0.0 || row.melodynePitchCenterCents > 0.0
        ? juce::jlimit(0.0f, 127.0f, storedSource) : -1.0f;
    note.midiNote = row.melodynePitchCenterCents > 0.0
        ? juce::jlimit(0.0f, 127.0f, static_cast<float>(row.melodynePitchCenterCents / 100.0))
        : juce::jlimit(0.0f, 127.0f, storedSource + static_cast<float>(row.relativePitchCents / 100.0));
    note.drift = juce::jlimit(0.0f, 2.0f, static_cast<float>(row.melodynePitchDrift));
    note.modulation = juce::jlimit(0.0f, 2.0f, static_cast<float>(row.melodynePitchModulation));
    note.formantSemitones = juce::jlimit(-12.0f, 12.0f, static_cast<float>(row.melodyneFormantCents / 100.0));
    note.breath = juce::jlimit(0.0f, 1.0f, static_cast<float>(row.melodyneSibilantBalance));
    note.gain = juce::jlimit(0.0f, 4.0f, static_cast<float>(row.melodyneAmplitude));
    note.attackSpeed = juce::jlimit(0.05f, 20.0f, static_cast<float>(row.melodyneAttackSeconds > 1.0e-6
        ? row.fixedDurationSeconds / row.melodyneAttackSeconds : 1.0));
    note.nativeSegments = SampleSettings::nativeSegmentsFor(row);
    note.amplitudeEnvelope = row.amplitudeEnvelope;
    note.utauPreutteranceOverrideEnabled = true;
    note.utauPreutteranceSeconds = std::max(0.0, row.alignmentSeconds - row.regionStartSeconds);
    note.utauOverlapOverrideEnabled = std::abs(row.overlapSeconds) > 1.0e-9;
    note.utauOverlapSeconds = row.overlapSeconds;
    // A source centre in metadata is not sampled F0. Analysis can replace this
    // explicitly unmeasured neutral stub without mistaking it for real data.
    note.sourcePitchMeasured = false;
    note.contour = {{0.0, 0.0f, 0.0f, !note.nativeUnpitched},
                    {note.durationSeconds, 0.0f, 0.0f, !note.nativeUnpitched}};
    if (note.nativeUnpitched)
    {
        note.sourceMidiCenter = note.midiNote;
        note.sourcePitchMeasured = legacyUnpitchedMeasured;
        note.nativeIndependentPitch = true; note.utauAutoPitchTransition = false;
    }
}
}

juce::String ProjectModel::addAudioFile(const juce::File& file, double durationSeconds,
                                        double startSeconds,
                                        const juce::String& targetTrackId,
                                        const VoicebankOtoEntry* oto, UtauMode otoMode)
{
    ClipData clip;
    clip.id = makeId("clip");
    clip.sourceFile = file;
    clip.startSeconds = std::max(0.0, startSeconds);
    clip.durationSeconds = std::max(0.01, durationSeconds);
    clip.sourceDurationSeconds = clip.durationSeconds;

    // Voicebank registration writes timing beside the source before the file
    // is dragged into a project.  Consume that sidecar immediately so an OTO
    // sample arrives as editable note objects instead of a silent/empty piano
    // roll that requires drawing every region again.
    const auto current = snapshot();
    const auto accompaniment = std::any_of(current.tracks.begin(), current.tracks.end(),
        [&](const auto& track) { return track.id == targetTrackId && track.accompaniment; });
    juce::String annotationError;
    const auto nativeAnnotated = SampleSettings::hasNativeAnnotation(file, &annotationError);
    if (annotationError.isNotEmpty() && oto == nullptr) return {};
    if (oto != nullptr)
    {
        if (oto->audioFile != file || accompaniment) return {};
        for (const auto& track : current.tracks)
            if (track.id == targetTrackId && !trackShowsAllNativeRegions(track)) return {};
        auto prepared = SampleSettings::nativeAudioFromOto(*oto, otoMode, durationSeconds);
        if (!prepared) return {};
        prepared->id = clip.id;
        prepared->startSeconds = clip.startSeconds;
        clip = std::move(*prepared);
        for (auto& note : clip.notes) note.id = makeId("note");
    }
    else if (!accompaniment && nativeAnnotated)
    {
        const auto rows = SampleSettings::loadOrDerive(file, ProjectData{}, &annotationError);
        if (annotationError.isNotEmpty()) return {};
        std::vector<RegionSpan> spans;
        spans.reserve(rows.size());
        for (const auto& row : rows)
        {
            const auto regionStart = juce::jlimit(0.0, clip.durationSeconds,
                                                   row.regionStartSeconds);
            const auto regionEnd = juce::jlimit(regionStart, clip.durationSeconds,
                                                 row.regionEndSeconds);
            spans.push_back({ regionStart,
                              regionEnd - regionStart < 0.001 ? regionStart : regionEnd });
        }
        for (const auto index : regionsToImport(spans))
        {
            const auto& row = rows[index];
            const auto regionStart = spans[index].startSeconds;
            const auto regionEnd = spans[index].endSeconds;
            if (regionEnd - regionStart < 0.001) continue;
            NoteData note;
            note.id = makeId("note");
            note.startSeconds = regionStart;
            note.durationSeconds = regionEnd - regionStart;
            initialiseMaterialNote(note, row, true);
            clip.notes.push_back(std::move(note));
        }
    }
    const auto clipId = clip.id;

    {
        const juce::ScopedLock guard(lock);
        pushUndoLocked();
        const auto target = std::find_if(project.tracks.begin(), project.tracks.end(),
            [&targetTrackId](const auto& track)
            {
                return targetTrackId.isNotEmpty() && track.id == targetTrackId;
            });
        if (target != project.tracks.end())
        {
            if (!trackShowsAllNativeRegions(*target))
                std::erase_if(clip.notes, [](const auto& n) { return n.nativeUnpitched; });
            if (trackIsDiffSinger(*target)) for (auto& note : clip.notes) note.utauFlagCurveEnabled = true;
            auto relative=clip;relative.startSeconds=0;clip.startSeconds=nativePasteStart(*target,{relative},clip.startSeconds);
            target->clips.push_back(std::move(clip));
        }
        else
        {
            TrackData track;configureNewTrackType(track,TrackType::hachi);
            track.id = makeId("track");
            track.name = file.getFileNameWithoutExtension();
            track.clips.push_back(std::move(clip));
            project.tracks.push_back(std::move(track));
        }
        if (project.name == "Untitled")
            project.name = file.getFileNameWithoutExtension();
    }
    sendChangeMessage();
    return clipId;
}

juce::String ProjectModel::addMaterialPronunciations(
    const juce::File& audio, const std::vector<SampleRegionSetting>& completeRows,
    const std::vector<juce::String>& selectedRegionIds, double absoluteStart,
    const juce::String& targetTrackId, juce::String& error, const juce::String& chosenAlias)
{
    error.clear();
    const auto fail = [&](const juce::String& message) -> juce::String { error = message; return {}; };
    if (!std::isfinite(absoluteStart) || absoluteStart < 0)
        return fail("Invalid material insertion time.");
    if (completeRows.empty() || selectedRegionIds.empty()) return fail("No material pronunciations selected.");
    // Reuse the manifest authority, including scoped identities, finite source
    // metadata, alias uniqueness within a row, explicit links and cycle checks.
    OttoMaterialIO::Document document;
    document.audio.push_back({completeRows.front().materialAudioId, audio.getFileName(), completeRows});
    if (!OttoMaterialIO::validate(document, error)) return {};
    juce::AudioFormatManager formats; formats.registerBasicFormats();
    std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(audio));
    if (!reader || reader->sampleRate <= 0) return fail("Cannot read material source audio.");
    const auto sourceDuration = reader->lengthInSamples / reader->sampleRate;
    constexpr double epsilon = 1.0e-8;
    for (const auto& row : completeRows)
        if (row.regionEndSeconds > sourceDuration + epsilon)
            return fail("Material pronunciation lies outside source audio: " + row.materialRegionId);
    std::vector<const SampleRegionSetting*> selected;
    for (const auto& id : selectedRegionIds)
    {
        if (std::any_of(selected.begin(), selected.end(), [&](const auto* row) { return row->materialRegionId == id; }))
            return fail("Duplicate selected pronunciation identity: " + id);
        const auto found = std::find_if(completeRows.begin(), completeRows.end(),
            [&](const auto& row) { return row.materialRegionId == id; });
        if (found == completeRows.end()) return fail("Unknown selected pronunciation identity: " + id);
        if (found->regionEndSeconds - found->regionStartSeconds < .001)
            return fail("Pronunciation shorter than the native 1ms minimum: " + id);
        selected.push_back(&*found);
    }
    std::stable_sort(selected.begin(), selected.end(), [](const auto* a, const auto* b)
    { return a->regionStartSeconds < b->regionStartSeconds; });
    for (std::size_t i = 1; i < selected.size(); ++i)
        if (selected[i]->regionStartSeconds < selected[i-1]->regionEndSeconds - epsilon)
            return fail("Overlapping pronunciation selections require separate placements.");

    if(chosenAlias.isNotEmpty()&&(selected.size()!=1
        ||(chosenAlias!=selected.front()->name&&!selected.front()->additionalAliases.contains(chosenAlias))))
        return fail("Chosen alias must belong to the single selected pronunciation.");
    const auto firstSource = selected.front()->regionStartSeconds;
    const auto lastSource = selected.back()->regionEndSeconds;
    ClipData clip; clip.id = makeId("clip"); clip.sourceFile = audio;
    clip.startSeconds = absoluteStart;
    clip.durationSeconds = clip.sourceDurationSeconds = lastSource - firstSource;
    clip.sourceOffsetSeconds = firstSource;
    // An explicit identity map preserves authored source geometry rather than
    // letting consonant expression derive a second implicit warp on insertion.
    clip.sourceTimeMap = {{0, 0}, {clip.durationSeconds, clip.sourceDurationSeconds}};
    struct Run { std::size_t first, last; };
    std::vector<Run> runs;
    for (std::size_t i = 0; i < selected.size(); ++i)
    {
        NoteData note; note.id = makeId("note");
        note.startSeconds = selected[i]->regionStartSeconds - firstSource;
        note.durationSeconds = selected[i]->regionEndSeconds - selected[i]->regionStartSeconds;
        initialiseMaterialNote(note, *selected[i]);
        if(chosenAlias.isNotEmpty())
        {note.label=note.materialBoundAlias=chosenAlias;note.materialAliasFollowsPrimary=chosenAlias==selected[i]->name;}
        clip.notes.push_back(std::move(note));
        if (i == 0 || selected[i]->regionStartSeconds > selected[i-1]->regionEndSeconds + epsilon)
            runs.push_back({i, i});
        else runs.back().last = i;
    }
    // Discontinuous runs never share an audible source window. The parent
    // keeps original-source spacing; only children are expanded for rendering.
    if (runs.size() > 1)
    {
        clip.sourceTimeMap.clear();
        for (const auto& run : runs)
        {
            ClipData part; part.id = makeId("material-part"); part.sourceFile = audio;
            const auto begin = selected[run.first]->regionStartSeconds;
            const auto end = selected[run.last]->regionEndSeconds;
            part.startSeconds = begin - firstSource; part.sourceOffsetSeconds = begin;
            part.durationSeconds = part.sourceDurationSeconds = end - begin;
            part.sourceTimeMap = {{0, 0}, {part.durationSeconds, part.sourceDurationSeconds}};
            for (auto i = run.first; i <= run.last; ++i) clip.notes[i].clipPartId = part.id;
            clip.parts.push_back(std::move(part));
        }
    }
    std::vector<NativeConnection> connections;
    for (std::size_t i = 1; i < selected.size(); ++i)
    {
        const auto& left = *selected[i-1]; const auto& right = *selected[i];
        if (left.connectedNextRegionId != right.materialRegionId) continue;
        // Only explicit links with consecutive selected endpoints become
        // project connections. The codec already checked <=1ms continuity.
        if (std::abs(left.regionEndSeconds - right.regionStartSeconds)
            > OttoMaterialIO::connectionToleranceSeconds + epsilon) continue;
        clip.notes[i-1].connectedToNext = true; clip.notes[i].connectedToPrevious = true;
        connections.push_back({makeId("connection"), clip.notes[i-1].id, clip.notes[i].id,
            "pitch-and-amplitude", (left.regionEndSeconds + right.regionStartSeconds) * .5 - firstSource,
            {}, {}});
        if (clip.notes[i-1].clipPartId != clip.notes[i].clipPartId)
            for (auto& part : clip.parts)
            {
                if (part.id == clip.notes[i-1].clipPartId) part.glideConnectedToNext = true;
                if (part.id == clip.notes[i].clipPartId) part.glideConnectedFromPrevious = true;
            }
    }
    if (!clip.parts.empty())
        clip.nativeAudioLinked = std::all_of(clip.parts.begin() + 1, clip.parts.end(),
            [](const auto& part) { return part.glideConnectedFromPrevious; });

    const auto clipId = clip.id;
    {
        const juce::ScopedLock guard(lock);
        auto target = std::find_if(project.tracks.begin(), project.tracks.end(),
            [&](const auto& track) { return targetTrackId.isNotEmpty()
                && track.id == targetTrackId && trackShowsAllNativeRegions(track); });
        if (target != project.tracks.end())
        {
            auto relative = clip; relative.startSeconds = 0;
            clip.startSeconds = nativePasteStart(*target, {relative}, absoluteStart);
            for (auto& connection : connections) connection.boundarySeconds += clip.startSeconds;
            pushUndoLocked();
            target->clips.push_back(std::move(clip));
        }
        else
        {
            TrackData track;configureNewTrackType(track,TrackType::hachi); track.id = makeId("track"); track.name = audio.getFileNameWithoutExtension();
            track.compose = true; track.nativeNsfAudio = true;
            // A newly created native track keeps the explicit native defaults.
            for (auto& connection : connections) connection.boundarySeconds += clip.startSeconds;
            track.clips.push_back(std::move(clip));
            pushUndoLocked();
            project.tracks.push_back(std::move(track));
        }
        project.nativeConnections.insert(project.nativeConnections.end(), connections.begin(), connections.end());
        if (project.name == "Untitled") project.name = audio.getFileNameWithoutExtension();
    }
    sendChangeMessage();
    return clipId;
}

juce::String ProjectModel::addTypedTrack(TrackType type,const juce::String& name)
{
    return addTrack(name,true,false,false,type);
}

juce::String ProjectModel::addTrack(const juce::String& requestedName, bool compose,
                                   bool referenceOnly, bool accompaniment,TrackType type)
{
    juce::String id;
    {
        const juce::ScopedLock guard(lock);
        pushUndoLocked();
        TrackData track;
        track.id = makeId("track");
        track.name = requestedName.trim().substring(0, 80);
        if (track.name.isEmpty())
            track.name = accompaniment ? "Accompaniment" : compose ? "Melodic Track" : "Audio Track";
        track.accompaniment = accompaniment;
        track.compose = compose && !accompaniment;
        track.referenceOnly = referenceOnly && !accompaniment;
        configureNewTrackType(track,accompaniment?TrackType::hachi:type);
        // New native tracks always use NSF and splice-before-synthesis defaults;
        // previously saved tracks retain their own explicit configuration.
        id = track.id;
        project.tracks.push_back(std::move(track));
    }
    sendChangeMessage();
    return id;
}

void ProjectModel::setTrackReferenceOnly(const juce::String& trackId,
                                         bool referenceOnly)
{
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            if (track.id == trackId && !track.accompaniment && track.referenceOnly != referenceOnly)
            {
                pushUndoLocked();
                track.referenceOnly = referenceOnly;
                changed = true;
            }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::setTrackName(const juce::String& trackId,
                                const juce::String& requestedName)
{
    const auto name = requestedName.trim().substring(0, 80);
    if (name.isEmpty()) return;
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            if (track.id == trackId && track.name != name)
            {
                pushUndoLocked();
                track.name = name;
                changed = true;
                break;
            }
    }
    if (changed) sendChangeMessage();
}

bool ProjectModel::setClipNotesIfEmpty(const juce::String& clipId,
                                       std::vector<NoteData> notes)
{
    if (notes.empty()) return false;
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
        {
            for (auto& clip : track.clips)
                if (!track.accompaniment && clip.id == clipId && clip.notes.empty())
                {
                    // Import analysis is one operation.  More importantly,
                    // do not replace notes the user drew while it was running.
                    const auto ratio = clip.durationSeconds > 1.0e-9
                        ? clip.sourceDurationSeconds / clip.durationSeconds : 1.0;
                    for (auto& note : notes)
                    {
                        if (trackIsDiffSinger(track)) note.utauFlagCurveEnabled = true;
                        if (note.label.trim().isEmpty()) note.label = "-";
                        note.nativeRole = note.label == "_"
                            ? NativeSegmentRole::transition : NativeSegmentRole::unknown;
                        note.nativeProvenance = note.nativeUnpitched ? "uncovered-audio" : "estimated";
                        note.nativeConfidence = 0.0f;
                        note.nativeSourceStartSeconds = clip.sourceOffsetSeconds
                            + note.startSeconds * ratio;
                        note.nativeSourceEndSeconds = note.nativeSourceStartSeconds
                            + note.durationSeconds * ratio;
                        note.nativeSegments = { { "segment_1", note.label,
                            note.nativeRole, 0.0, note.durationSeconds,
                            "estimated", 0.0f, note.durationSeconds, 0.0,
                            true, 1.0 } };
                    }
                    clip.notes = std::move(notes);
                    changed = true;
                    break;
                }
            if (changed) break;
        }
    }
    // Analysis supplies project notes, never authored source pronunciations.
    // Source OTTO is published only by an explicit native annotation save.
    if (changed) sendChangeMessage();
    return changed;
}

bool ProjectModel::attachSourceAnalysisLocked(const juce::File& file,const std::vector<NoteData>& sourceNotes,
                                               const juce::String& guardedFingerprint)
{
    const auto reference=nativeSourcePitchReference(sourceNotes,[](double t){return t;});
    if(reference->empty())return false;
    bool changed=false;
    for(auto& track:project.tracks)if(trackShowsAllNativeRegions(track))for(auto& parent:track.clips)
    {
        std::vector<ClipData*> owners;
        if(parent.parts.empty())owners.push_back(&parent);else for(auto& p:parent.parts)owners.push_back(&p);
        for(auto* owner:owners)
        {
            if(owner->sourceFile!=file)continue;
            const bool guarded=owner->materialSourceFingerprint.isNotEmpty();
            if(guarded&&(guardedFingerprint.isEmpty()||guardedFingerprint!=owner->materialSourceFingerprint))continue;
            if(guarded&&!owner->materialSourceAnalysisPending&&!owner->nativeSourcePitchPending)continue;
            auto proxy=*owner;proxy.parts.clear();proxy.notes.clear();
            const auto offset=parent.parts.empty()?0.0:owner->startSeconds;
            for(auto n:parent.notes)if(parent.parts.empty()||n.clipPartId==owner->id)
            {n.startSeconds-=offset;proxy.notes.push_back(std::move(n));}
            if(guarded&&proxy.notes.empty())continue; // never resurrect deleted source audio
            const bool pending=owner->nativeSourcePitchPending;
            if(guarded)
            {
                // The one acoustic mapper is reused, but its placeholder-pitch
                // adoption must NEVER become a source-replacement score edit.
                auto measured=proxy;
                if(backend::AnalysisService::applySourcePitch(measured,sourceNotes)==0)continue;
                proxy.nativeSourcePitch=reference;proxy.nativeSourcePitchComplete=true;
                const auto clock=nativeClipClock(proxy);
                for(auto& live:parent.notes)if(parent.parts.empty()||live.clipPartId==owner->id)
                {
                    const auto found=std::find_if(measured.notes.begin(),measured.notes.end(),[&](const auto& n){return n.id==live.id;});
                    if(found==measured.notes.end())continue;
                    const auto score=ownPitchPoints(live); // CURRENT edits, not the ticket's old snapshot
                    std::vector<double> times{0,live.durationSeconds};
                    for(const auto& p:found->contour)times.push_back(p.timeSeconds);
                    for(const auto& p:live.contour)if(p.hasManualTarget&&p.timeSeconds>0&&p.timeSeconds<live.durationSeconds)times.push_back(p.timeSeconds);
                    for(const auto& p:score)if(p.timeSeconds>0&&p.timeSeconds<live.durationSeconds)times.push_back(p.timeSeconds);
                    std::sort(times.begin(),times.end());times.erase(std::unique(times.begin(),times.end()),times.end());
                    std::vector<PitchPoint> facts;const auto centre=found->sourceMidiCenter>=0?found->sourceMidiCenter:live.midiNote;
                    for(auto t:times)
                    {
                        const auto source=owner->sourceOffsetSeconds+nativeSourceTimeAt(clock,live.startSeconds-offset+t);
                        const auto ref=nativeReferencePitchAt(proxy,source);PitchPoint p;p.timeSeconds=t;
                        p.voiced=!live.nativeUnpitched&&ref&&ref->voiced;
                        if(p.voiced){p.relativeCents=(ref->midi-centre)*100;p.withoutVibratoCents=(ref->withoutVibratoMidi-centre)*100;}
                        p.hasManualTarget=true;p.manualTargetCents=(evaluatePitchCurve(score,t)-live.midiNote)*100;
                        facts.push_back(p);
                    }
                    live.contour=std::move(facts);live.sourceMidiCenter=found->sourceMidiCenter;
                    live.sourcePitchMeasured=true; // only real full-file facts reach this branch
                }
                owner->muted=owner->materialSourceResumeMuted;owner->materialSourceAnalysisPending=false;
            }
            else if(pending)
            {
                proxy.nativeSourcePitch=reference;proxy.nativeSourcePitchComplete=true;
                for(auto& n:proxy.notes)refreshNativeTrimPitch(n,proxy);
                for(auto& live:parent.notes)
                    if(const auto n=std::find_if(proxy.notes.begin(),proxy.notes.end(),[&](const auto& n){return n.id==live.id;});n!=proxy.notes.end())
                    {live.contour=n->contour;live.sourceMidiCenter=n->sourceMidiCenter;live.sourcePitchMeasured=n->sourcePitchMeasured;}
            }
            owner->nativeSourcePitch=reference;owner->nativeSourcePitchComplete=true;owner->nativeSourcePitchPending=false;
            changed=true;
        }
    }
    return changed;
}

bool ProjectModel::setNativeTrimSourceAnalysis(const juce::File& file,const std::vector<NoteData>& sourceNotes,
                                               const juce::String& expectedSourceFingerprint)
{
    if(sourceNotes.empty())return false;
    if(expectedSourceFingerprint.isNotEmpty()&&(!file.existsAsFile()||juce::SHA256(file).toHexString()!=expectedSourceFingerprint))return false;
    bool changed=false;
    {
        const juce::ScopedLock guard(lock);
        if(expectedSourceFingerprint.isNotEmpty())
        {
            materialSourceAnalysisCache[expectedSourceFingerprint]=std::make_shared<const std::vector<NoteData>>(sourceNotes);
            while(materialSourceAnalysisCache.size()>8)materialSourceAnalysisCache.erase(materialSourceAnalysisCache.begin());
        }
        changed=attachSourceAnalysisLocked(file,sourceNotes,expectedSourceFingerprint);
    }
    if(changed)sendChangeMessage();return changed;
}

std::vector<ProjectModel::MaterialSourceBindingTicket> ProjectModel::pendingMaterialSourceAnalyses() const
{
    const juce::ScopedLock guard(lock);std::vector<MaterialSourceBindingTicket> result;
    for(const auto& track:project.tracks)for(const auto& parent:track.clips)
    {
        const auto add=[&](const ClipData& owner,bool root)
        {
            if(!owner.materialSourceAnalysisPending||owner.materialSourceFingerprint.isEmpty())return;
            const auto note=std::find_if(parent.notes.begin(),parent.notes.end(),[&](const auto& n){return root||n.clipPartId==owner.id;});
            if(note!=parent.notes.end())result.push_back({note->id,parent.id,owner.id,owner.sourceFile,owner.materialSourceFingerprint});
        };
        if(parent.parts.empty())add(parent,true);else for(const auto& owner:parent.parts)add(owner,false);
    }
    return result;
}

bool ProjectModel::completeMaterialSourceAnalysis(const MaterialSourceBindingTicket& ticket,
    const std::vector<NoteData>& fullFileSourceNotes,juce::String& error)
{
    error.clear();
    if(ticket.noteId.isEmpty()||ticket.parentClipId.isEmpty()||ticket.sourcePartId.isEmpty()||ticket.sourceFingerprint.isEmpty())
    {error="Incomplete source binding ticket";return false;}
    if(!ticket.sourceFile.existsAsFile()||juce::SHA256(ticket.sourceFile).toHexString()!=ticket.sourceFingerprint)
    {error="Source audio changed while acoustic analysis was pending; reselect the recording.";return false;}
    for(const auto& n:fullFileSourceNotes)
    {
        if(!std::isfinite(n.startSeconds)||!std::isfinite(n.durationSeconds)||n.durationSeconds<=0||!std::isfinite(n.midiNote)||!std::isfinite(n.sourceMidiCenter))
        {error="Invalid acoustic source analysis";return false;}
        double previous=-std::numeric_limits<double>::infinity();
        for(const auto& p:n.contour)
        {
            if(!std::isfinite(p.timeSeconds)||p.timeSeconds<previous||!std::isfinite(p.relativeCents)||!std::isfinite(p.withoutVibratoCents))
            {error="Invalid acoustic source contour";return false;}
            previous=p.timeSeconds;
        }
    }
    bool changed=false;
    {
        const juce::ScopedLock guard(lock);
        std::shared_ptr<const std::vector<NoteData>> facts;
        if(!fullFileSourceNotes.empty())
        {
            facts=std::make_shared<const std::vector<NoteData>>(fullFileSourceNotes);
            materialSourceAnalysisCache[ticket.sourceFingerprint]=facts;
            while(materialSourceAnalysisCache.size()>8)materialSourceAnalysisCache.erase(materialSourceAnalysisCache.begin());
        }
        else if(const auto cached=materialSourceAnalysisCache.find(ticket.sourceFingerprint);cached!=materialSourceAnalysisCache.end())facts=cached->second;
        if(!facts){error="Full-file source analysis is required (cache miss); reschedule the pending ticket.";return false;}
        // The ticket identifies immutable FILE facts, not permission to restore
        // its old note/source/score. Only current matching owners can consume it.
        changed=attachSourceAnalysisLocked(ticket.sourceFile,*facts,ticket.sourceFingerprint);
    }
    if(changed)sendChangeMessage();return changed;
}

std::size_t ProjectModel::markNativeUnpitchedRegions(const juce::String& clipId)
{
    std::size_t added = 0;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks) if (trackShowsAllNativeRegions(track))
            for (auto& clip : track.clips) if (clip.id == clipId)
            {
                auto updated = clip;
                if (clip.parts.empty()) added = appendNativeUnpitchedRegions(updated);
                else for (auto part : expandedClipParts(clip))
                {
                    const auto before = part.notes;
                    added += appendNativeUnpitchedRegions(part);
                    for (auto note : part.notes)
                        if (std::none_of(before.begin(), before.end(), [&](const auto& n) { return n.id == note.id; }))
                        {
                            note.startSeconds += part.startSeconds - clip.startSeconds;
                            note.clipPartId = part.id.substring(clip.id.length() + 1);
                            updated.notes.push_back(std::move(note));
                        }
                }
                if (added) { pushUndoLocked(); clip = std::move(updated); }
                break;
            }
    }
    if (added) sendChangeMessage();
    return added;
}

bool ProjectModel::setClipAudioAnalysis(const juce::String& clipId,
                                        std::vector<NoteData> notes,
                                        const ClipData& importedClip)
{
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
            {
                if (clip.id != clipId || track.accompaniment || !clip.parts.empty()) continue;
                const auto& imported = importedClip.id.isEmpty() ? clip : importedClip;
                if (clip.sourceFile != imported.sourceFile) continue;
                const auto native = !trackUsesVoicebankSynthesis(track) && !trackIsDiffSinger(track);
                if (imported.notes.empty())
                {
                    // Analysis cannot replace notes drawn while it was pending.
                    if (!clip.notes.empty()) return false;
                    if (!native) return setClipNotesIfEmpty(clipId, std::move(notes));
                    clip.nativeSourcePitch=nativeSourcePitchReference(notes,[](double t){return t;});
                    clip.nativeSourcePitchComplete=true;
                    const auto audio = nativeAudioPreviewClip(clip);
                    const auto map = nativeSourceTimeMap(audio);
                    const auto sourceEnd = audio.sourceOffsetSeconds + audio.sourceDurationSeconds;
                    auto mapped = clip;
                    for (auto note : notes)
                    {
                        const auto first = std::max(audio.sourceOffsetSeconds, note.startSeconds);
                        const auto last = std::min(sourceEnd, note.startSeconds + note.durationSeconds);
                        if (last - first < 0.001) continue;
                        const auto targetStart = nativeTargetTimeAt(map, first - audio.sourceOffsetSeconds);
                        const auto targetEnd = nativeTargetTimeAt(map, last - audio.sourceOffsetSeconds);
                        const auto attackEnd = nativeTargetTimeAt(map, juce::jlimit(0.0,
                            audio.sourceDurationSeconds, note.startSeconds + note.consonantSeconds
                                - audio.sourceOffsetSeconds));
                        const auto sourceAttack = std::max(0.0,
                            std::min(last, note.startSeconds + note.consonantSeconds) - first);
                        note.startSeconds = clip.audioStartSeconds + targetStart;
                        note.durationSeconds = std::max(0.001, targetEnd - targetStart);
                        note.consonantSeconds = juce::jlimit(0.0, note.durationSeconds, attackEnd - targetStart);
                        if (note.consonantSeconds > 1.0e-9)
                            note.attackSpeed = juce::jlimit(0.05f, 20.0f,
                                static_cast<float>(sourceAttack / note.consonantSeconds));
                        note.contour.clear();
                        mapped.notes.push_back(std::move(note));
                    }
                    appendNativeUnpitchedRegions(mapped);
                    backend::AnalysisService::applySourcePitch(mapped, notes);
                    for (auto& note : mapped.notes)
                    {
                        note.midiNote = note.sourceMidiCenter;
                        // Acoustic breath evidence describes the original
                        // recording; it is not a request to add more breath.
                        note.breath = 0.0f;
                    }
                    return setClipNotesIfEmpty(clipId, std::move(mapped.notes));
                }
                if (!native) return false;
                auto measured = clip;
                std::erase_if(measured.notes, [&](const auto& note)
                {
                    return std::none_of(imported.notes.begin(), imported.notes.end(),
                        [&](const auto& original) { return original.id == note.id; });
                });
                if (backend::AnalysisService::applySourcePitch(measured, notes) == 0) return false;
                clip.nativeSourcePitch=measured.nativeSourcePitch;
                clip.nativeSourcePitchComplete=measured.nativeSourcePitchComplete;
                for (auto& note : clip.notes)
                    if (const auto found = std::find_if(measured.notes.begin(), measured.notes.end(),
                        [&](const auto& measuredNote) { return measuredNote.id == note.id; });
                        found != measured.notes.end())
                    {
                        note.sourceMidiCenter = found->sourceMidiCenter;
                        note.sourcePitchMeasured = found->sourcePitchMeasured;
                        note.midiNote = found->midiNote;
                        note.contour = std::move(found->contour);
                    }
                // Do not refill gaps in a clip already changed by the user.
                if (clip.startSeconds == imported.startSeconds && clip.durationSeconds == imported.durationSeconds
                    && clip.sourceOffsetSeconds == imported.sourceOffsetSeconds
                    && clip.sourceDurationSeconds == imported.sourceDurationSeconds
                    && clip.notes.size() == imported.notes.size())
                    appendNativeUnpitchedRegions(clip);
                changed = true;
                break;
            }
    }
    if (changed) sendChangeMessage();
    return changed;
}

bool ProjectModel::writeMidiFile(const ProjectData& data, const juce::File& file,
                                 juce::String& error, const juce::String& trackId,
                                 bool includeMuted)
{
    error.clear();
    if (trackId.isNotEmpty() && std::none_of(data.tracks.begin(), data.tracks.end(),
        [&](const auto& track) { return track.id == trackId; }))
    {
        error = "unknown MIDI export track: " + trackId;
        return false;
    }
    // Ticks, because that is what a MIDI file counts in and what keeps the
    // song on its beats wherever it is opened.  480 to the quarter is what a
    // UST counts in as well, so a note written as a whole number of ticks
    // there comes back out as one here.
    constexpr auto ticksPerQuarter = 480;
    const auto tickOf = [&data](double seconds)
    {
        // Tick zero is the project's first beat.  Anything before it -- a
        // note dragged ahead of the timeline's origin -- has nowhere to go in
        // a MIDI file, which starts where it starts.
        return juce::jmax(0, juce::roundToInt(
            data.quarterPositionForSeconds(seconds) * ticksPerQuarter));
    };

    juce::MidiFile midi;
    midi.setTicksPerQuarterNote(ticksPerQuarter);

    // The conductor track: what the song is called, how fast it goes and how
    // it is counted.  Its own track, as a type 1 file wants it.
    juce::MidiMessageSequence conductor;
    conductor.addEvent(juce::MidiMessage::textMetaEvent(3, data.name), 0.0);
    conductor.addEvent(juce::MidiMessage::timeSignatureMetaEvent(
        juce::jmax(1, data.numerator), juce::jmax(1, data.denominator)), 0.0);
    const auto tempoEvent = [](double bpm)
    {
        return juce::MidiMessage::tempoMetaEvent(juce::roundToInt(
            60.0e6 / juce::jlimit(20.0, 400.0, bpm)));
    };
    // The speed, and every change of it, at one event per tick: a change
    // written at the very start is the song's speed rather than a second
    // answer beside it -- which is what a UST whose first note carries a
    // tempo of its own leaves behind.  Readers that take the first of two
    // would play the whole song at a speed nothing is sung at.
    std::vector<std::pair<double, double>> speeds { { 0.0, data.bpm } };
    for (const auto& change : data.tempoChanges)
        speeds.emplace_back(juce::jmax(0.0, change.quarterPosition) * ticksPerQuarter,
                            change.bpm);
    std::stable_sort(speeds.begin(), speeds.end(),
                     [](const auto& left, const auto& right)
                     { return left.first < right.first; });
    for (std::size_t index = 0; index < speeds.size(); ++index)
    {
        if (index + 1 < speeds.size()
            && std::abs(speeds[index + 1].first - speeds[index].first) < 1.0e-9)
            continue;
        conductor.addEvent(tempoEvent(speeds[index].second), speeds[index].first);
    }
    conductor.updateMatchedPairs();
    midi.addTrack(conductor);

    auto notesWritten = 0;
    for (const auto& track : data.tracks)
    {
        if ((trackId.isNotEmpty() && track.id != trackId) || (!includeMuted && track.muted)) continue;
        struct Written
        {
            int start = 0, end = 0, number = 0, velocity = 100;
            juce::String lyric;
        };
        std::vector<Written> notes;
        for (const auto& clip : track.clips)
        {
            if (!includeMuted && clip.muted) continue;
            for (const auto& note : clip.notes)
            {
                if (note.nativeUnpitched || backend::isRestLyric(note.label)) continue;
                Written written;
                written.start = tickOf(clip.startSeconds + note.startSeconds);
                written.end = juce::jmax(written.start + 1,
                    tickOf(clip.startSeconds + note.startSeconds + note.durationSeconds));
                written.number = juce::jlimit(0, 127, juce::roundToInt(note.midiNote));
                // A note's loudness here is a multiplier of its own level and
                // is 1 almost everywhere; velocity 100 is what a UST calls
                // that, so the two agree about what "as written" means.
                written.velocity = juce::jlimit(1, 127,
                    juce::roundToInt(note.gain * 100.0f));
                written.lyric = note.label.trim();
                notes.push_back(std::move(written));
            }
        }
        if (notes.empty()) continue;
        std::stable_sort(notes.begin(), notes.end(),
                         [](const Written& left, const Written& right)
                         {
                             if (left.start != right.start) return left.start < right.start;
                             return left.number < right.number;
                         });
        // Two notes of the same pitch that overlap are one note-off short of
        // each other: whichever off arrives first ends both, and the second
        // hangs to the end of the song.  The earlier one gives way.
        for (std::size_t index = 0; index + 1 < notes.size(); ++index)
            for (auto later = index + 1; later < notes.size(); ++later)
            {
                if (notes[later].start >= notes[index].end) break;
                if (notes[later].number != notes[index].number) continue;
                notes[index].end = juce::jmax(notes[index].start + 1, notes[later].start);
                break;
            }

        juce::MidiMessageSequence sequence;
        sequence.addEvent(juce::MidiMessage::textMetaEvent(3, track.name), 0.0);
        for (const auto& note : notes)
        {
            // The lyric sits with the note it is sung on, which is where every
            // reader of a singing MIDI looks for it.
            if (note.lyric.isNotEmpty())
                sequence.addEvent(juce::MidiMessage::textMetaEvent(5, note.lyric),
                                  note.start);
            sequence.addEvent(juce::MidiMessage::noteOn(1, note.number,
                static_cast<juce::uint8>(note.velocity)), note.start);
            sequence.addEvent(juce::MidiMessage::noteOff(1, note.number), note.end);
            ++notesWritten;
        }
        sequence.updateMatchedPairs();
        midi.addTrack(sequence);
    }

    if (notesWritten == 0)
    {
        error = trackId.isEmpty() ? "this project has no exportable notes (muted notes may be excluded)"
                                  : "the selected track has no exportable notes (muted notes may be excluded)";
        return false;
    }
    juce::TemporaryFile temporary(file);
    {
        // Written beside the target and moved into place, so a write that
        // fails partway leaves the file that was there untouched.
        std::unique_ptr<juce::FileOutputStream> stream(
            temporary.getFile().createOutputStream());
        if (stream == nullptr || !midi.writeTo(*stream))
        {
            error = "could not write the MIDI file: " + file.getFullPathName();
            return false;
        }
    }
    if (!temporary.overwriteTargetFileWithTemporary())
    {
        error = "could not write the MIDI file: " + file.getFullPathName();
        return false;
    }
    return true;
}

bool ProjectModel::addMidiFile(const juce::File& file, juce::String& error)
{
    auto input = file.createInputStream();
    if (input == nullptr)
    {
        error = "Could not open MIDI file: " + file.getFullPathName();
        return false;
    }
    juce::MidiFile midi;
    if (!midi.readFrom(*input))
    {
        error = "Invalid MIDI file: " + file.getFullPathName();
        return false;
    }
    std::optional<double> importedBpm;
    for (int trackIndex = 0; trackIndex < midi.getNumTracks() && !importedBpm; ++trackIndex)
        if (const auto* sequence = midi.getTrack(trackIndex))
            for (int eventIndex = 0; eventIndex < sequence->getNumEvents(); ++eventIndex)
                if (const auto* event = sequence->getEventPointer(eventIndex);
                    event != nullptr && event->message.isTempoMetaEvent())
                {
                    const auto secondsPerQuarter = event->message.getTempoSecondsPerQuarterNote();
                    if (secondsPerQuarter > 1.0e-9)
                        importedBpm = 60.0 / secondsPerQuarter;
                    break;
                }
    midi.convertTimestampTicksToSeconds();
    std::vector<TrackData> importedTracks;
    for (int trackIndex = 0; trackIndex < midi.getNumTracks(); ++trackIndex)
    {
        const auto* sequence = midi.getTrack(trackIndex);
        if (sequence == nullptr) continue;
        juce::MidiMessageSequence matched(*sequence);
        matched.updateMatchedPairs();
        TrackData track;configureNewTrackType(track,TrackType::hachi);
        track.id = makeId("track");
        track.name = file.getFileNameWithoutExtension()
            + (midi.getNumTracks() > 1 ? " " + juce::String(trackIndex + 1) : juce::String());
        track.compose = true;
        ClipData clip;
        clip.id = makeId("clip");
        clip.sourceFile = file;
        clip.startSeconds = 0.0;
        for (int eventIndex = 0; eventIndex < matched.getNumEvents(); ++eventIndex)
        {
            const auto* event = matched.getEventPointer(eventIndex);
            if (event == nullptr || !event->message.isNoteOn()) continue;
            const auto offIndex = matched.getIndexOfMatchingKeyUp(eventIndex);
            const auto start = std::max(0.0, event->message.getTimeStamp());
            const auto end = offIndex >= 0
                ? std::max(start + 0.01, matched.getEventTime(offIndex)) : start + 0.25;
            NoteData note;
            note.id = makeId("note");
            note.startSeconds = start;
            note.durationSeconds = end - start;
            note.consonantSeconds = 0.0;
            note.midiNote = static_cast<float>(event->message.getNoteNumber());
            note.sourceMidiCenter = note.midiNote;
            note.contour.push_back({ 0.0, 0.0f, 0.0f, true });
            note.contour.push_back({ note.durationSeconds, 0.0f, 0.0f, true });
            clip.durationSeconds = std::max(clip.durationSeconds, end);
            clip.notes.push_back(std::move(note));
        }
        if (clip.notes.empty()) continue;
        clip.sourceDurationSeconds = clip.durationSeconds;
        track.clips.push_back(std::move(clip));
        importedTracks.push_back(std::move(track));
    }
    if (importedTracks.empty())
    {
        error = "MIDI file contains no notes: " + file.getFullPathName();
        return false;
    }
    {
        const juce::ScopedLock guard(lock);
        pushUndoLocked();
        if (project.tracks.empty() && importedBpm)
            project.bpm = juce::jlimit(20.0, 400.0, *importedBpm);
        for (auto& track : importedTracks) project.tracks.push_back(std::move(track));
        if (project.name == "Untitled") project.name = file.getFileNameWithoutExtension();
    }
    sendChangeMessage();
    return true;
}

namespace
{
// A MIDI file as one-track import reads it.
//
// Positions are in quarter notes, the file's own musical time, so the notes
// can be put on the project's beats whatever its tempo.  A file timed in SMPTE
// frames has no beats: its positions are seconds, and say so.
struct MidiPart
{
    int index = 0;
    juce::String name;
    std::vector<NoteData> notes;   // start and duration in quarters, or seconds
};
struct MidiSong
{
    std::vector<MidiPart> parts;   // the tracks with notes, in file order
    std::optional<double> bpm;
    std::vector<TempoChange> tempoChanges;
    std::optional<std::pair<int, int>> meter;
    bool inSeconds = false;
};

juce::MemoryBlock metaEventBytes(const juce::MidiMessage& message)
{
    return juce::MemoryBlock(message.getMetaEventData(),
                             static_cast<size_t>(juce::jmax(0, message.getMetaEventLength())));
}

// A track's name and its lyrics, read in one encoding.  A MIDI file says
// nothing about how its text is written: a Japanese one is usually Shift-JIS,
// a Chinese one GBK, a newer one UTF-8.  Read together they are read the way
// a UST is -- valid UTF-8 as UTF-8, otherwise the reading with the most kana --
// and one lyric cannot come out in a different encoding from the next.
juce::StringArray decodeMidiTexts(const std::vector<juce::MemoryBlock>& texts)
{
    const auto clean = [](const juce::MemoryBlock& bytes)
    {
        juce::MemoryBlock kept;
        const auto* data = static_cast<const char*>(bytes.getData());
        for (size_t index = 0; index < bytes.getSize(); ++index)
            if (data[index] != '\n' && data[index] != '\r' && data[index] != 0)
                kept.append(data + index, 1);
        return kept;
    };
    juce::MemoryBlock joined;
    for (size_t index = 0; index < texts.size(); ++index)
    {
        if (index > 0) joined.append("\n", 1);
        const auto kept = clean(texts[index]);
        joined.append(kept.getData(), kept.getSize());
    }
    juce::String encoding;
    auto lines = juce::StringArray::fromTokens(backend::UstImporter::decode(joined, encoding),
                                               "\n", "");
    // No multi-byte encoding this can meet puts a line feed inside a
    // character, so the lines come back one for one; should they not, each is
    // read on its own rather than handed to the wrong note.
    if (lines.size() != static_cast<int>(texts.size()))
    {
        lines.clear();
        for (const auto& text : texts)
            lines.add(backend::UstImporter::decode(clean(text), encoding));
    }
    for (auto& line : lines) line = line.trim();
    return lines;
}

std::optional<MidiSong> readMidiSong(const juce::File& file, juce::String& error)
{
    auto input = file.createInputStream();
    if (input == nullptr)
    {
        error = "Could not open MIDI file: " + file.getFullPathName();
        return std::nullopt;
    }
    juce::MidiFile midi;
    if (!midi.readFrom(*input))
    {
        error = "Invalid MIDI file: " + file.getFullPathName();
        return std::nullopt;
    }
    MidiSong song;
    const auto ticksPerQuarter = static_cast<double>(midi.getTimeFormat());
    song.inSeconds = ticksPerQuarter <= 0.0;
    if (song.inSeconds) midi.convertTimestampTicksToSeconds();
    const auto positionOf = [&song, ticksPerQuarter](double stamp)
    {
        return juce::jmax(0.0, song.inSeconds ? stamp : stamp / ticksPerQuarter);
    };

    // The tempo map, from whichever tracks carry it: a type 1 file keeps it in
    // its conductor track, a type 0 file beside the notes.
    std::vector<std::pair<double, double>> speeds;
    for (int trackIndex = 0; trackIndex < midi.getNumTracks(); ++trackIndex)
        if (const auto* sequence = midi.getTrack(trackIndex))
            for (int eventIndex = 0; eventIndex < sequence->getNumEvents(); ++eventIndex)
            {
                const auto& message = sequence->getEventPointer(eventIndex)->message;
                if (message.isTempoMetaEvent() && message.getTempoSecondsPerQuarterNote() > 1.0e-9)
                    speeds.emplace_back(positionOf(message.getTimeStamp()),
                                        juce::jlimit(20.0, 400.0,
                                                     60.0 / message.getTempoSecondsPerQuarterNote()));
                else if (message.isTimeSignatureMetaEvent() && !song.meter)
                {
                    int numerator = 4, denominator = 4;
                    message.getTimeSignatureInfo(numerator, denominator);
                    song.meter = std::make_pair(numerator, denominator);
                }
            }
    std::stable_sort(speeds.begin(), speeds.end(),
                     [](const auto& left, const auto& right) { return left.first < right.first; });
    if (!song.inSeconds && !speeds.empty())
    {
        // Until the first change a MIDI file runs at 120.  Of two changes at
        // one place the later is the one that holds.
        song.bpm = speeds.front().first <= 1.0e-9 ? speeds.front().second : 120.0;
        for (std::size_t index = 0; index < speeds.size(); ++index)
        {
            if (index + 1 < speeds.size() && speeds[index + 1].first - speeds[index].first < 1.0e-9)
                continue;
            if (speeds[index].first <= 1.0e-9) song.bpm = speeds[index].second;
            else song.tempoChanges.push_back({ speeds[index].first, speeds[index].second });
        }
    }

    for (int trackIndex = 0; trackIndex < midi.getNumTracks(); ++trackIndex)
    {
        const auto* sequence = midi.getTrack(trackIndex);
        if (sequence == nullptr) continue;
        juce::MidiMessageSequence matched(*sequence);
        matched.updateMatchedPairs();
        MidiPart part;
        part.index = trackIndex;
        std::vector<juce::MemoryBlock> texts(1);   // the name first, then the lyrics
        auto named = false;
        std::optional<juce::MemoryBlock> pendingLyric;
        std::vector<std::pair<std::size_t, std::size_t>> lyricOf;   // note, text
        for (int eventIndex = 0; eventIndex < matched.getNumEvents(); ++eventIndex)
        {
            const auto& message = matched.getEventPointer(eventIndex)->message;
            if (message.isTrackNameEvent())
            {
                if (!named) texts.front() = metaEventBytes(message);
                named = true;
                continue;
            }
            // A lyric is sung on the note it comes before, or with: the next
            // one to start.
            if (message.isMetaEvent() && message.getMetaEventType() == 5)
            {
                pendingLyric = metaEventBytes(message);
                continue;
            }
            if (!message.isNoteOn()) continue;
            const auto offIndex = matched.getIndexOfMatchingKeyUp(eventIndex);
            const auto start = positionOf(message.getTimeStamp());
            const auto shortest = song.inSeconds ? 0.01 : 1.0 / 64.0;
            const auto end = offIndex >= 0
                ? juce::jmax(start + shortest, positionOf(matched.getEventTime(offIndex)))
                : start + (song.inSeconds ? 0.25 : 0.5);
            NoteData note;
            note.startSeconds = start;
            note.durationSeconds = end - start;
            note.consonantSeconds = 0.0;
            note.midiNote = static_cast<float>(message.getNoteNumber());
            note.sourceMidiCenter = note.midiNote;
            if (pendingLyric)
            {
                lyricOf.emplace_back(part.notes.size(), texts.size());
                texts.push_back(std::move(*pendingLyric));
                pendingLyric.reset();
            }
            part.notes.push_back(std::move(note));
        }
        if (part.notes.empty()) continue;
        const auto decoded = decodeMidiTexts(texts);
        part.name = decoded[0];
        for (const auto& [noteIndex, textIndex] : lyricOf)
            part.notes[noteIndex].label = decoded[static_cast<int>(textIndex)];
        song.parts.push_back(std::move(part));
    }
    if (song.parts.empty())
    {
        error = "MIDI file contains no notes: " + file.getFullPathName();
        return std::nullopt;
    }
    return song;
}
}

std::vector<ProjectModel::MidiTrackChoice> ProjectModel::midiTrackChoices(const juce::File& file,
                                                                          juce::String& error)
{
    std::vector<MidiTrackChoice> choices;
    if (const auto song = readMidiSong(file, error))
        for (const auto& part : song->parts)
            choices.push_back({ part.index, part.name, static_cast<int>(part.notes.size()) });
    return choices;
}

juce::String ProjectModel::addMidiTrack(const juce::File& file, int trackIndex,
                                        juce::String& error, const juce::String& targetTrackId, double startSeconds)
{
    if (!std::isfinite(startSeconds) || startSeconds < 0.0)
    {
        error = "Invalid MIDI insertion position.";
        return {};
    }
    auto song = readMidiSong(file, error);
    if (!song) return {};
    const auto part = std::find_if(song->parts.begin(), song->parts.end(),
                                   [trackIndex](const MidiPart& each) { return each.index == trackIndex; });
    if (part == song->parts.end())
    {
        error = "MIDI track " + juce::String(trackIndex + 1) + " has no notes: "
            + file.getFullPathName();
        return {};
    }

    TrackData track;configureNewTrackType(track,TrackType::hachi);
    track.id = makeId("track");
    track.name = part->name.isNotEmpty() ? part->name
        : file.getFileNameWithoutExtension()
              + (song->parts.size() > 1 ? " " + juce::String(trackIndex + 1) : juce::String());
    track.name = track.name.substring(0, 80);
    track.compose = true;
    ClipData clip;
    clip.id = makeId("clip");
    clip.sourceFile = file;
    clip.startSeconds = targetTrackId.isEmpty() ? 0.0 : startSeconds;
    clip.notes = std::move(part->notes);

    const auto id = targetTrackId.isEmpty() ? track.id : targetTrackId;
    {
        const juce::ScopedLock guard(lock);
        TrackData* destination = nullptr;
        if (targetTrackId.isNotEmpty())
        {
            for (auto& existing : project.tracks)
                if (existing.id == targetTrackId) { destination = &existing; break; }
            if (destination == nullptr || !destination->compose || destination->accompaniment || destination->referenceOnly)
            {
                error = "The destination melodic track is no longer available.";
                return {};
            }
        }
        pushUndoLocked();
        // A new native MIDI track keeps NSF + splice-before-synthesis defaults.
        // Adopt the song tempo only when no existing score can disagree.
        if (project.tracks.empty() && !song->inSeconds)
        {
            if (song->bpm) project.bpm = *song->bpm;
            project.tempoChanges = song->tempoChanges;
            if (song->meter)
            {
                project.numerator = juce::jlimit(1, 32, song->meter->first);
                const auto denominator = song->meter->second;
                project.denominator = denominator == 2 || denominator == 8 || denominator == 16
                    ? denominator : 4;
            }
        }
        const auto startQuarter = destination != nullptr
            ? project.quarterPositionForSeconds(clip.startSeconds) : 0.0;
        for (auto& note : clip.notes)
        {
            note.id = makeId("note");
            if (!song->inSeconds)
            {
                const auto startQuarters = startQuarter + note.startSeconds;
                const auto endQuarters = startQuarters + note.durationSeconds;
                const auto absoluteStart = project.secondsForQuarterPosition(startQuarters);
                note.startSeconds = absoluteStart - clip.startSeconds;
                note.durationSeconds = std::max(0.01,
                    project.secondsForQuarterPosition(endQuarters) - absoluteStart);
            }
            note.contour.push_back({ 0.0, 0.0f, 0.0f, true });
            note.contour.push_back({ note.durationSeconds, 0.0f, 0.0f, true });
            clip.durationSeconds = std::max(clip.durationSeconds,
                                            note.startSeconds + note.durationSeconds);
        }
        std::stable_sort(clip.notes.begin(), clip.notes.end(),
            [](const auto& left, const auto& right) { return left.startSeconds < right.startSeconds; });
        clip.sourceDurationSeconds = clip.durationSeconds;
        if (destination != nullptr)
            destination->clips.push_back(std::move(clip));
        else
        {
            track.clips.push_back(std::move(clip));
            project.tracks.push_back(std::move(track));
            if (project.name == "Untitled") project.name = file.getFileNameWithoutExtension();
        }
    }
    sendChangeMessage();
    return id;
}

namespace
{
// A UST mode-2 pitch bend as this editor's pitch handles.
//
// Every pitch in a UST bend is in tenths of a semitone and is measured from
// the note's own NoteNum, so a note at 63 whose bend starts at -20 starts two
// semitones down, at 61 -- which is how a portamento out of the preceding
// note is written.  Times are milliseconds: PBS gives the first point's
// offset from the note's start (normally negative, so the bend begins inside
// the note before), and PBW gives the gap to each point after it.
//
// PBY is one short of PBW on purpose: the last point is the note's own pitch.
//
// PBM names the shape of each segment, and is indexed by the segment rather
// than the point -- entry i shapes the run from point i to point i+1, which is
// this editor's "shape of the incoming segment" on point i+1.
// A UST amplitude envelope as this editor's envelope handles.
//
// UTAU measures the shape from the beginning of the rendered output, which
// starts one preutterance before the note; this editor measures from the note
// itself, so time zero here is one preutterance later and the opening ramp
// sits at negative times.  That is the same convention envelopeForNoteAsItIs
// re-anchors at render time, and it is what keeps p1 lined up with the
// crossfade into the previous note.
//
// Volumes are percentages of the note's own level.  Zero is silence, and the
// renderer treats anything at or below -60 dB as silent, so that is the floor
// rather than an infinity.
void applyUstEnvelope(NoteData& note, const backend::UstNote& source,
                      double preutteranceSeconds)
{
    if (!source.hasEnvelope) return;
    const auto gainDb = [](double percent)
    {
        if (percent <= 0.01) return -60.0f;
        return juce::jlimit(-60.0f, 12.0f,
                            static_cast<float>(20.0 * std::log10(percent / 100.0)));
    };
    const auto fromStart = [preutteranceSeconds](double milliseconds)
    {
        return -preutteranceSeconds + milliseconds / 1000.0;
    };
    const auto fromEnd = [&note](double milliseconds)
    {
        return note.durationSeconds - milliseconds / 1000.0;
    };

    std::vector<AmplitudeEnvelopePoint> points;
    points.push_back({ fromStart(0.0), -60.0f });
    points.push_back({ fromStart(source.envelopeP1), gainDb(source.envelopeV1) });
    points.push_back({ fromStart(source.envelopeP1 + source.envelopeP2),
                       gainDb(source.envelopeV2) });
    if (source.hasMiddlePoint)
        points.push_back({ fromStart(source.envelopeP1 + source.envelopeP2
                                     + source.envelopeP5),
                           gainDb(source.envelopeV5) });
    points.push_back({ fromEnd(source.envelopeP3 + source.envelopeP4),
                       gainDb(source.envelopeV3) });
    points.push_back({ fromEnd(source.envelopeP3), gainDb(source.envelopeV4) });
    points.push_back({ fromEnd(0.0), -60.0f });

    // A short note can leave the opening ramp and the closing one overlapping,
    // and an envelope whose times run backwards is not one this editor can
    // draw or the renderer can read.  Push each point up to the one before it
    // rather than dropping any: the shape stays, squeezed.
    for (std::size_t index = 1; index < points.size(); ++index)
        points[index].timeSeconds = std::max(points[index].timeSeconds,
                                             points[index - 1].timeSeconds);
    for(auto& point:points)point.linearToNext=true;
    note.amplitudeEnvelope = std::move(points);
}

void applyUstPitchBend(NoteData& note, const backend::UstNote& source)
{
    if (!source.hasPitchBend) return;
    const auto pitchAt = [&note](double tenths)
    {
        return note.midiNote + static_cast<float>(tenths) * 0.1f;
    };
    const auto shapeFor = [&source](int segment)
    {
        const auto name = segment < source.shapes.size()
            ? source.shapes[segment] : juce::String();
        if (name == "s") return PitchCurveShape::linear;
        // UTAU's R rises fast and flattens; its J waits and then rises.
        if (name == "r") return PitchCurveShape::easeOut;
        if (name == "j") return PitchCurveShape::easeIn;
        // An empty entry is UTAU's default S-curve.
        return PitchCurveShape::smooth;
    };

    std::vector<PitchCurveEditPoint> points;
    points.reserve(source.widthsMs.size() + 1);
    auto time = source.pitchStartMs / 1000.0;
    points.push_back({ time, pitchAt(source.pitchStartTenths) });
    for (std::size_t index = 0; index < source.widthsMs.size(); ++index)
    {
        time += source.widthsMs[index] / 1000.0;
        const auto tenths = index < source.pitchTenths.size()
            ? source.pitchTenths[index] : 0.0;
        PitchCurveEditPoint point { time, pitchAt(tenths) };
        point.shape = shapeFor(static_cast<int>(index));
        points.push_back(point);
    }
    // A bend with one point says nothing the note's own pitch does not.
    if (points.size() < 2) return;
    note.pitchControlPoints = std::move(points);
}
}

bool ProjectModel::addUstFile(const juce::File& file, juce::String& error,
                              juce::StringArray& warnings, UstImportMode mode,
                              juce::String* importedTrackId, int encoding,TrackType destinationType)
{
    const auto parsed = backend::UstImporter::read(file, error, warnings, encoding);
    if (!parsed) return false;

    struct PartView{juce::String name,voiceDirectory,globalFlags;const std::vector<backend::UstNote>* notes;};
    std::vector<PartView> parts{{parsed->name,parsed->voiceDirectory,parsed->globalFlags,&parsed->notes}};
    for(const auto& part:parsed->additionalTracks)parts.push_back({part.name,part.voiceDirectory,part.globalFlags,&part.notes});
    struct PreparedPart{TrackData track;ClipData clip;std::vector<const backend::UstNote*> sources;const std::vector<backend::UstNote>* sourceNotes;double quarters;};
    std::vector<PreparedPart> prepared;std::vector<TempoChange> tempoChanges;
    for(const auto& tempo:parsed->tempoChanges)
    {
        if(!std::isfinite(tempo.quarterPosition)||tempo.quarterPosition<0||!std::isfinite(tempo.bpm)||tempo.bpm<20||tempo.bpm>400)
        {error="The score tempo map is outside the supported 20..400 BPM range.";return false;}
        tempoChanges.push_back({tempo.quarterPosition,tempo.bpm});
    }
    if(parsed->sourceFormat=="ustx"&&(!std::isfinite(parsed->tempo)||parsed->tempo<20||parsed->tempo>400))
    {error="The score tempo is outside the supported 20..400 BPM range.";return false;}
    if(parsed->sourceFormat=="ustx"&&(parsed->numerator<1||parsed->numerator>32
        ||(parsed->denominator!=2&&parsed->denominator!=4&&parsed->denominator!=8&&parsed->denominator!=16)))
    {error="The score meter cannot be represented by this project; no notes were imported.";return false;}
    std::stable_sort(tempoChanges.begin(),tempoChanges.end(),[](const auto& x,const auto& y){return x.quarterPosition<y.quarterPosition;});
    for(std::size_t partIndex=0;partIndex<parts.size();++partIndex)
    {
    const auto& part=parts[partIndex];
    TrackData track;configureNewTrackType(track,destinationType);
    track.id = makeId("track");
    track.name = part.name.isNotEmpty() ? part.name
                                           : file.getFileNameWithoutExtension();
    track.compose = true;
    // Destination family is an explicit import choice, defaulting to Hachi.
    // Musical/UST metadata below is retained independently of that choice.
    track.utauGlobalFlags = part.globalFlags;
    track.ustSourceDocument=parsed->sourceText; track.ustSourceEncoding=parsed->sourceEncoding; track.ustSourceBom=parsed->sourceBom; track.ustSourceBytes=parsed->sourceBytes;

    ClipData clip;
    clip.id = makeId("clip");
    clip.sourceFile = file;
    clip.startSeconds = 0.0;

    // A UST places its notes by accumulating lengths in ticks, so the timeline
    // is built in musical time and turned into seconds through the project's
    // own tempo map afterwards.  Summing seconds as we went would drift the
    // moment a note carried a tempo change.
    std::vector<const backend::UstNote*> sources;
    auto quarters = 0.0;
    auto tempo = parsed->tempo;
    for (const auto& source : *part.notes)
    {
        const auto position=source.positionQuarterNotes.value_or(quarters);
        if(!std::isfinite(position)||position<0){error="Score positions must be finite and nonnegative.";return false;}
        if (partIndex==0 && parsed->tempoChanges.empty() && source.tempo && *source.tempo > 0.0 && *source.tempo != tempo)
        {
            tempo = *source.tempo;
            tempoChanges.push_back({ position, juce::jlimit(20.0, 400.0, tempo) });
        }
        const auto length = source.durationQuarterNotes.value_or(backend::UstImporter::quarterNotes(source.lengthTicks));
        if(!std::isfinite(length)||(source.durationQuarterNotes&&length<=0)){error="Invalid score duration.";return false;}
        // A rest becomes a gap rather than a silent note, which is the shape
        // the rest of this editor already works in.
        if (source.isRest() || length <= 0.0)
        {
            quarters = std::max(quarters,position+length);
            continue;
        }
        NoteData note;
        note.id = makeId("note");
        note.label = source.lyric.trim();
        note.nativeRole = note.label == "_" || note.label.containsChar('_')
            ? NativeSegmentRole::transition
            : note.label == "-" ? NativeSegmentRole::unknown
            : NativeSegmentRole::vowel;
            note.nativeProvenance = "utau";
            note.nativeConfidence = note.nativeRole == NativeSegmentRole::unknown ? 0.0f : 1.0f;
        note.midiNote = static_cast<float>(source.noteNum);
        note.sourceMidiCenter = note.midiNote;
        note.utauFlags = source.flags;
        note.ustSourceSection=source.sourceSection; note.ustSourceSectionIndex=source.sourceSectionIndex;
        note.utauModulationPercent=source.modulation.value_or(0);
        note.utauStpSeconds=source.stpMs.value_or(0)/1000.0;
        // A UST states every note's pitch, bend or no bend.  Without one the
        // note is on its own pitch throughout, and nothing of the editor's is
        // laid over either.
        note.utauAutoPitchTransition = false;
        if(destinationType==TrackType::diffSinger)note.utauFlagCurveEnabled=true;
        // The vibrato as the file writes it; the editor's own vibrato is the
        // same seven numbers, so they are carried across as they stand.
        if (source.mode2 && source.hasVibrato)
        {
            note.vibratoEnabled = true;
            note.vibratoLengthPercent = juce::jlimit(0.0, 100.0, source.vibratoLengthPercent);
            note.vibratoCycleMs = std::max(1.0, source.vibratoCycleMs);
            note.vibratoDepthCents = source.vibratoDepthCents;
            note.vibratoFadeInPercent = juce::jlimit(0.0, 100.0, source.vibratoFadeInPercent);
            note.vibratoFadeOutPercent = juce::jlimit(0.0, 100.0, source.vibratoFadeOutPercent);
            note.vibratoPhasePercent = juce::jlimit(-100.0, 100.0, source.vibratoPhasePercent);
            note.vibratoOffsetPercent = juce::jlimit(-100.0, 100.0, source.vibratoOffsetPercent);
        }
        if (source.velocity) note.utauConsonantVelocity = *source.velocity;
        // UST intensity is a percentage; this editor keeps gain as a
        // multiplier and the renderer turns it back into a percentage.
        if (source.intensity)
            note.gain = juce::jlimit(0.0f, 2.0f,
                                     static_cast<float>(*source.intensity) / 100.0f);
        // An absent PreUtterance means "whatever the oto says", which is not
        // the same as an override of zero, so only a written value overrides.
        if (source.preutteranceMs)
        {
            note.utauPreutteranceOverrideEnabled = true;
            note.utauPreutteranceSeconds = *source.preutteranceMs / 1000.0;
        }
        if (source.overlapMs)
        {
            note.utauOverlapOverrideEnabled = true;
            note.utauOverlapSeconds = *source.overlapMs / 1000.0;
        }
        // Musical position for now; converted below, once the tempo map is in.
        note.startSeconds = position;
        note.durationSeconds = length;
        quarters = std::max(quarters,position+length);
        clip.notes.push_back(std::move(note));
        sources.push_back(&source);
    }

    if(clip.notes.empty()){warnings.add("Empty/rest-only voice track omitted: "+part.name);continue;}
    prepared.push_back({std::move(track),std::move(clip),std::move(sources),part.notes,quarters});
    }
    if(prepared.empty()){error="The score contains no sounding notes: "+file.getFullPathName();return false;}
    const auto newTrackId=prepared.front().track.id;
    {
        const juce::ScopedLock guard(lock);
        pushUndoLocked();
        if(mode==UstImportMode::replaceProject)
        {const auto visible=project.assetManagerVisible;project=ProjectData{};project.assetManagerVisible=visible;}
        if(project.tracks.empty())
        {
            project.bpm=juce::jlimit(20.0,400.0,parsed->tempo);project.tempoChanges=tempoChanges;
            project.numerator=juce::jlimit(1,32,parsed->numerator);
            project.denominator=parsed->denominator==2||parsed->denominator==8||parsed->denominator==16?parsed->denominator:4;
        }
        else if(!tempoChanges.empty())warnings.add("tempo changes ignored: the project already has tracks");
        for(auto& part:prepared)
        {
        auto& track=part.track;auto& clip=part.clip;auto& sources=part.sources;
        for (std::size_t index = 0; index < clip.notes.size(); ++index)
        {
            auto& note = clip.notes[index];
            const auto startQuarters = note.startSeconds;
            const auto endQuarters = startQuarters + note.durationSeconds;
            note.startSeconds = project.secondsForQuarterPosition(startQuarters);
            const auto seconds=project.secondsForQuarterPosition(endQuarters)-note.startSeconds;
            note.durationSeconds = sources[index]->durationQuarterNotes ? seconds : std::max(0.01,seconds);
            note.consonantSeconds = 0.0;
            note.nativeSegments = { { "segment_1", note.label.isEmpty() ? "-" : note.label,
                note.nativeRole, 0.0, note.durationSeconds, "utau",
                note.nativeConfidence, note.durationSeconds, note.utauOverlapSeconds,
                note.nativeRole != NativeSegmentRole::consonant, 1.0 } };
            note.contour.push_back({ 0.0, 0.0f, 0.0f, true });
            note.contour.push_back({ note.durationSeconds, 0.0f, 0.0f, true });
            if(parsed->mode2)applyUstPitchBend(note, *sources[index]);
            else backend::ustexchange::applyMode1(note,*sources[index],project.tempoAtSeconds(note.startSeconds),sources[index]>part.sourceNotes->data()?sources[index]-1:nullptr);
            // The envelope is drawn against the preutterance the UST named.
            // Where it named none the oto's own is not known here -- no
            // voicebank has been chosen yet -- so the shape is anchored at the
            // note and envelopeForNoteAsItIs moves it, ramps intact, once the
            // real preutterance is known at render time.
            applyUstEnvelope(note, *sources[index],
                             note.utauPreutteranceOverrideEnabled
                                 ? note.utauPreutteranceSeconds : 0.0);
            note.ustBaseline=juce::JSON::toString(backend::ustexchange::noteState(note,project,clip.startSeconds,parsed->mode2),true);
        }
        clip.durationSeconds=project.secondsForQuarterPosition(part.quarters);
        for(const auto& note:clip.notes)clip.durationSeconds=std::max(clip.durationSeconds,note.startSeconds+note.durationSeconds);
        clip.sourceDurationSeconds = clip.durationSeconds;
        track.clips.push_back(std::move(clip));
        ProjectData originalTempoProject;originalTempoProject.bpm=parsed->tempo;originalTempoProject.tempoChanges=tempoChanges;
        auto ustTrackBaseline=backend::ustexchange::trackState(track,project);
        const auto originalTempoState=backend::ustexchange::trackState(track,originalTempoProject);
        ustTrackBaseline.getDynamicObject()->setProperty("Tempo",originalTempoState["Tempo"]);
        ustTrackBaseline.getDynamicObject()->setProperty("_tempos",originalTempoState["_tempos"]);
        track.ustBaseline=juce::JSON::toString(ustTrackBaseline,true);
        project.tracks.push_back(std::move(track));
        }
        if(project.name=="Untitled"&&parsed->name.isNotEmpty())project.name=parsed->name;
    }
    if(importedTrackId)*importedTrackId=newTrackId;
    for(const auto& part:parts)if(part.voiceDirectory.isNotEmpty())warnings.add("This score asks for voicebank \""+part.voiceDirectory+"\"; configure it explicitly for the chosen track type.");
    sendChangeMessage();return true;
}

bool ProjectModel::exportUst(const juce::File& file,const juce::String& trackId,juce::String& error,juce::StringArray& warnings,int encoding) const
{ return backend::ustexchange::write(snapshot(),trackId,file,error,warnings,encoding); }

void ProjectModel::setTempo(double bpm, int numerator, int denominator)
{
    {
        const juce::ScopedLock guard(lock);
        pushUndoLocked();
        project.bpm = juce::jlimit(20.0, 400.0, bpm);
        project.numerator = juce::jlimit(1, 32, numerator);
        project.denominator = denominator == 2 || denominator == 8 || denominator == 16
            ? denominator : 4;
    }
    sendChangeMessage();
}

void ProjectModel::setTempoChange(double quarterPosition, double bpm, bool synchronizeNotes)
{
    const auto position = std::max(0.0, quarterPosition);
    const auto tempo = juce::jlimit(20.0, 400.0, bpm);
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        const auto before = project;
        auto existing = std::find_if(project.tempoChanges.begin(),
            project.tempoChanges.end(), [&](const auto& change)
            {
                return std::abs(change.quarterPosition - position) < 1.0e-7;
            });
        if (position <= 1.0e-7)
        {
            if (std::abs(project.bpm - tempo) < 1.0e-7) return;
        }
        else if (existing != project.tempoChanges.end()
                 && std::abs(existing->bpm - tempo) < 1.0e-7)
            return;

        pushUndoLocked();
        if (position <= 1.0e-7)
            project.bpm = tempo;
        else if (existing != project.tempoChanges.end())
            existing->bpm = tempo;
        else
            project.tempoChanges.push_back({ position, tempo });
        std::stable_sort(project.tempoChanges.begin(), project.tempoChanges.end(),
            [](const auto& left, const auto& right)
            {
                return left.quarterPosition < right.quarterPosition;
            });

        const auto remapTime = [&](double oldSeconds)
        {
            return project.secondsForQuarterPosition(
                before.quarterPositionForSeconds(oldSeconds));
        };
        for (std::size_t trackIndex = 0; trackIndex < project.tracks.size(); ++trackIndex)
        {
            auto& track = project.tracks[trackIndex];
            if (!synchronizeNotes || !track.compose || trackIndex >= before.tracks.size()) continue;
            const auto& oldTrack = before.tracks[trackIndex];
            for (std::size_t clipIndex = 0;
                 clipIndex < track.clips.size() && clipIndex < oldTrack.clips.size();
                 ++clipIndex)
            {
                auto& clip = track.clips[clipIndex];
                const auto& oldClip = oldTrack.clips[clipIndex];
                const auto oldClipStart = oldClip.startSeconds;
                const auto oldClipEnd = oldClipStart + oldClip.durationSeconds;
                const auto newClipStart = remapTime(oldClipStart);
                const auto newClipEnd = remapTime(oldClipEnd);
                clip.startSeconds = newClipStart;
                clip.durationSeconds = std::max(0.01, newClipEnd - newClipStart);

                const auto remapClipOffset = [&](double oldOffset)
                {
                    return remapTime(oldClipStart + oldOffset) - newClipStart;
                };
                for (auto& point : clip.sourceTimeMap)
                    point.targetSeconds = remapClipOffset(point.targetSeconds);
                for (auto& point : clip.nativeTrimClock)
                    point.targetSeconds = remapClipOffset(point.targetSeconds);
                const auto oldAudioStart = oldClipStart + oldClip.audioStartSeconds;
                const auto oldAudioEnd = oldAudioStart + oldClip.audioLength();
                const auto newAudioStart = remapTime(oldAudioStart);
                const auto newAudioEnd = remapTime(oldAudioEnd);
                if (oldClip.audioDurationSeconds >= 0.0)
                {
                    clip.audioStartSeconds = newAudioStart - newClipStart;
                    clip.audioDurationSeconds = std::max(0.0, newAudioEnd - newAudioStart);
                }
                clip.fadeInSeconds = juce::jlimit(0.0, clip.audioLength(),
                    remapTime(oldAudioStart + oldClip.fadeInSeconds) - newAudioStart);
                clip.fadeOutSeconds = juce::jlimit(0.0, clip.audioLength(),
                    newAudioEnd - remapTime(oldAudioEnd - oldClip.fadeOutSeconds));

                for(auto& part:clip.parts)
                {
                    const auto oldStart=oldClipStart+part.startSeconds;
                    const auto oldEnd=oldStart+part.durationSeconds;
                    const auto newStart=remapTime(oldStart);
                    const auto audioStart=oldStart+part.audioStartSeconds;
                    const auto audioEnd=audioStart+part.audioLength();
                    part.audioDurationSeconds=remapTime(audioEnd)-remapTime(audioStart);
                    part.audioStartSeconds=remapTime(audioStart)-newStart;
                    part.fadeInSeconds=remapTime(audioStart+part.fadeInSeconds)-remapTime(audioStart);
                    part.fadeOutSeconds=remapTime(audioEnd)-remapTime(audioEnd-part.fadeOutSeconds);
                    for(auto& point:part.sourceTimeMap)point.targetSeconds=remapTime(oldStart+point.targetSeconds)-newStart;
                    for(auto& point:part.nativeTrimClock)point.targetSeconds=remapTime(oldStart+point.targetSeconds)-newStart;
                    part.startSeconds=newStart-newClipStart;part.durationSeconds=remapTime(oldEnd)-newStart;
                }

                for (std::size_t noteIndex = 0;
                     noteIndex < clip.notes.size() && noteIndex < oldClip.notes.size();
                     ++noteIndex)
                {
                    auto& note = clip.notes[noteIndex];
                    const auto& oldNote = oldClip.notes[noteIndex];
                    const auto oldNoteStart = oldClipStart + oldNote.startSeconds;
                    const auto oldNoteEnd = oldNoteStart + oldNote.durationSeconds;
                    const auto newNoteStart = remapTime(oldNoteStart);
                    const auto newNoteEnd = remapTime(oldNoteEnd);
                    note.startSeconds = newNoteStart - newClipStart;
                    note.durationSeconds = std::max(0.001,
                                                    newNoteEnd - newNoteStart);
                    note.consonantSeconds = juce::jlimit(0.0, note.durationSeconds,
                        remapTime(oldNoteStart + oldNote.consonantSeconds)
                            - newNoteStart);
                    for (std::size_t index = 0;
                         index < note.contour.size() && index < oldNote.contour.size(); ++index)
                        note.contour[index].timeSeconds = remapTime(oldNoteStart
                            + oldNote.contour[index].timeSeconds) - newNoteStart;
                    for (std::size_t index = 0;
                         index < note.pitchControlPoints.size()
                            && index < oldNote.pitchControlPoints.size(); ++index)
                        note.pitchControlPoints[index].timeSeconds = remapTime(oldNoteStart
                            + oldNote.pitchControlPoints[index].timeSeconds) - newNoteStart;
                    for (std::size_t index = 0;
                         index < note.diffSingerPitchReference.size()
                            && index < oldNote.diffSingerPitchReference.size(); ++index)
                        note.diffSingerPitchReference[index].timeSeconds = remapTime(oldNoteStart
                            + oldNote.diffSingerPitchReference[index].timeSeconds) - newNoteStart;
                    for (std::size_t index = 0;
                         index < note.diffSingerPitchOffset.size()
                            && index < oldNote.diffSingerPitchOffset.size(); ++index)
                        note.diffSingerPitchOffset[index].timeSeconds = remapTime(oldNoteStart
                            + oldNote.diffSingerPitchOffset[index].timeSeconds) - newNoteStart;
                    mapDiffSingerParameterTimes(note,[&](double time){return remapTime(oldNoteStart+time)-newNoteStart;});
                    for (std::size_t index = 0;
                         index < note.amplitudeEnvelope.size()
                            && index < oldNote.amplitudeEnvelope.size(); ++index)
                        note.amplitudeEnvelope[index].timeSeconds = remapTime(oldNoteStart
                            + oldNote.amplitudeEnvelope[index].timeSeconds) - newNoteStart;
                    for (std::size_t index = 0;
                         index < note.sibilantMarkers.size()
                            && index < oldNote.sibilantMarkers.size(); ++index)
                        note.sibilantMarkers[index] = remapTime(oldNoteStart
                            + oldNote.sibilantMarkers[index]) - newNoteStart;
                    clip.durationSeconds = std::max(clip.durationSeconds,
                        note.startSeconds + note.durationSeconds);
                }
            }
        }
        changed = true;
    }
    if (changed) sendChangeMessage();
}

juce::int64 ProjectModel::contentFingerprint() const
{
    juce::MemoryOutputStream stream;
    toValueTree(juce::File(), false).writeToStream(stream);
    // Hash the raw bytes.  A serialised ValueTree is binary and full of zero
    // bytes, so reading it back as a string would stop at the first one and
    // leave almost the whole project out of the digest.
    const auto* bytes = static_cast<const juce::uint8*>(stream.getData());
    auto hash = 1469598103934665603ull;
    for (std::size_t index = 0; index < stream.getDataSize(); ++index)
    {
        hash ^= bytes[index];
        hash *= 1099511628211ull;
    }
    return static_cast<juce::int64>(hash);
}

void ProjectModel::setGridDivision(const juce::String& division)
{
    {
        const juce::ScopedLock guard(lock);
        pushUndoLocked();
        project.gridDivision = division;
    }
    sendChangeMessage();
}

void ProjectModel::setNoteEditDivision(int division)
{
    const auto value = juce::jlimit(2, 128, division);
    {
        const juce::ScopedLock guard(lock);
        if (project.noteEditDivision == value) return;
        pushUndoLocked();
        project.noteEditDivision = value;
    }
    sendChangeMessage();
}

void ProjectModel::setBaseScale(const juce::String& scale)
{
    {
        const juce::ScopedLock guard(lock);
        pushUndoLocked();
        project.baseScale = scale;
    }
    sendChangeMessage();
}

void ProjectModel::setTrackCompose(const juce::String& trackId, bool enabled)
{
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            if (track.id == trackId && !track.accompaniment && track.compose != enabled)
            {
                pushUndoLocked(); track.compose = enabled; changed = true; break;
            }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::setTrackMuted(const juce::String& trackId, bool muted)
{
    {
        const juce::ScopedLock guard(lock);
        pushUndoLocked();
        for (auto& track : project.tracks)
            if (track.id == trackId)
                track.muted = muted;
    }
    sendChangeMessage();
}

void ProjectModel::setTrackSolo(const juce::String& trackId, bool solo)
{
    {
        const juce::ScopedLock guard(lock);
        pushUndoLocked();
        for (auto& track : project.tracks)
            if (track.id == trackId)
                track.solo = solo;
    }
    sendChangeMessage();
}

void ProjectModel::setTrackVolume(const juce::String& trackId, float volume)
{
    {
        const juce::ScopedLock guard(lock);
        pushUndoLocked();
        for (auto& track : project.tracks)
            if (track.id == trackId)
                track.volume = juce::jlimit(0.0f, 2.0f, volume);
    }
    sendChangeMessage();
}

void ProjectModel::setClipGainEnvelope(const juce::String& clipId, std::vector<GainEnvelopePoint> points)
{
    points = normaliseTrackGainEnvelope(std::move(points));
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks) for (auto& clip : track.clips)
            if (clip.id == clipId)
            {
                if (sameTrackGainEnvelope(clip.gainEnvelope, points)) return;
                pushUndoLocked();
                clip.gainEnvelope = std::move(points);
                changed = true;
                break;
            }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::setTrackPan(const juce::String& trackId, float pan)
{
    {
        const juce::ScopedLock guard(lock);
        pushUndoLocked();
        for (auto& track : project.tracks)
            if (track.id == trackId)
                track.pan = juce::jlimit(-1.0f, 1.0f, pan);
    }
    sendChangeMessage();
}

void ProjectModel::setTrackSmoothOverlaps(const juce::String& trackId, bool enabled)
{
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            if (track.id == trackId && track.smoothOverlaps != enabled)
            {
                pushUndoLocked();
                track.smoothOverlaps = enabled;
                changed = true;
                break;
            }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::setTrackNormalizeVolume(const juce::String& trackId, bool enabled)
{
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            if (track.id == trackId && track.normalizeVolume != enabled)
            {
                pushUndoLocked();
                track.normalizeVolume = enabled;
                changed = true;
                break;
            }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::setTrackNsfSmoothPitchTransitions(const juce::String& trackId, bool enabled)
{
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            if (track.id == trackId && track.nsfSmoothPitchTransitions != enabled)
            {
                pushUndoLocked();
                track.nsfSmoothPitchTransitions = enabled;
                changed = true;
                break;
            }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::setTrackNsfNoiseProtection(const juce::String& trackId, bool enabled)
{
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            if (track.id == trackId && track.nsfNoiseProtection != enabled)
            {
                pushUndoLocked();
                track.nsfNoiseProtection = enabled;
                changed = true;
                break;
            }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::setTrackVoicebankDirectory(const juce::String& trackId,
                                               const juce::File& directory)
{
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            if (track.id == trackId && !track.accompaniment && track.voicebankDirectory != directory)
            {
                pushUndoLocked();
                if(!track.trackType)track.trackType=effectiveTrackType(track);
                track.voicebankDirectory = directory;
                // Bank contents are configuration, not a track-type conversion.
                // Selecting/moving a dsconfig file must not reinterpret music.
                changed = true;
                break;
            }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::setTrackUtauConsonantVelocity(const juce::String& trackId,
                                                  int velocity)
{
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            if (track.id == trackId && track.utauConsonantVelocity != velocity)
            {
                pushUndoLocked();
                track.utauConsonantVelocity = velocity;
                changed = true;
                break;
            }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::setTrackUtauGlobalFlags(const juce::String& trackId,
                                            const juce::String& flags)
{
    const auto normalized = flags.trim();
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            if (track.id == trackId && track.utauGlobalFlags != normalized)
            {
                pushUndoLocked();
                track.utauGlobalFlags = normalized;
                changed = true;
                break;
            }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::setTrackChineseCvvc(const juce::String& trackId, bool enabled)
{
    bool changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            if (track.id == trackId && track.chineseCvvc != enabled)
            {
                pushUndoLocked();
                track.chineseCvvc = enabled;
                changed = true;
                break;
            }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::setTrackPianoPreview(const juce::String& trackId, bool enabled)
{
    bool changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            if (track.id == trackId && track.utauPianoPreview != enabled)
            {
                pushUndoLocked();
                track.utauPianoPreview = enabled;
                changed = true;
                break;
            }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::setTrackCvvcCurveCrossfade(const juce::String& trackId, bool curved)
{
    bool changed=false;
    {
        const juce::ScopedLock guard(lock);
        for(auto& track:project.tracks)if(track.id==trackId&&track.cvvcCurveCrossfade!=curved)
        {pushUndoLocked();track.cvvcCurveCrossfade=curved;changed=true;break;}
    }
    if(changed)sendChangeMessage();
}

void ProjectModel::setPitchAlgorithm(PitchAlgorithm algorithm)
{
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            if (!track.accompaniment && (track.pitchAlgorithm != algorithm
                || (algorithm == PitchAlgorithm::nsfHifigan && !track.nativeNsfAudio)))
            {
                if (!changed) pushUndoLocked();
                const auto previousType=effectiveTrackType(track);
                track.pitchAlgorithm = algorithm;
                track.trackType=algorithm==PitchAlgorithm::utau?(previousType==TrackType::diffSinger?TrackType::diffSinger:TrackType::utau):TrackType::hachi;
                if (algorithm == PitchAlgorithm::nsfHifigan) track.nativeNsfAudio = true;
                changed = true;
            }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::setUtauMode(UtauMode mode)
{
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            if (!track.accompaniment && track.utauMode != mode)
            {
                if (!changed) pushUndoLocked();
                if(!track.trackType)track.trackType=effectiveTrackType(track);
                track.utauMode = effectiveTrackType(track)==TrackType::diffSinger?UtauMode::mou:mode;
                changed = true;
            }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::setTrackOutputEngine(const juce::String& trackId, UtauOutputEngine engine,
    const juce::File& resampler, const juce::File& wavtool)
{
    bool changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            if (track.id == trackId && trackUsesVoicebankSynthesis(track) && !trackIsDiffSinger(track))
            {
                if (track.outputEngine == engine && track.outputResampler == resampler && track.outputWavtool == wavtool) break;
                pushUndoLocked();
                track.outputEngine = engine;
                track.outputResampler = resampler;
                track.outputWavtool = wavtool;
                changed = true;
                break;
            }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::setTrackUtauMode(const juce::String& trackId, UtauMode mode)
{
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            if (track.id == trackId && !track.accompaniment && track.utauMode != mode)
            {
                pushUndoLocked();
                if(!track.trackType)track.trackType=effectiveTrackType(track);
                track.utauMode = effectiveTrackType(track)==TrackType::diffSinger?UtauMode::mou:mode;
                changed = true;
                break;
            }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::setTrackPitchAlgorithm(const juce::String& trackId,
                                          PitchAlgorithm algorithm)
{
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            if (track.id == trackId && !track.accompaniment && (track.pitchAlgorithm != algorithm
                || (algorithm == PitchAlgorithm::nsfHifigan && !track.nativeNsfAudio)))
            {
                pushUndoLocked();
                const auto previousType=effectiveTrackType(track);
                track.pitchAlgorithm = algorithm;
                track.trackType=algorithm==PitchAlgorithm::utau?(previousType==TrackType::diffSinger?TrackType::diffSinger:TrackType::utau):TrackType::hachi;
                if (algorithm == PitchAlgorithm::nsfHifigan) track.nativeNsfAudio = true;
                changed = true;
                break;
            }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::setStretchAlgorithm(StretchAlgorithm algorithm)
{
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            if (!track.accompaniment && track.stretchAlgorithm != algorithm)
            {
                if (!changed) pushUndoLocked();
                track.stretchAlgorithm = algorithm;
                changed = true;
            }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::setTrackStretchAlgorithm(const juce::String& trackId,
                                            StretchAlgorithm algorithm)
{
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            if (track.id == trackId && !track.accompaniment && track.stretchAlgorithm != algorithm)
            {
                pushUndoLocked();
                track.stretchAlgorithm = algorithm;
                changed = true;
                break;
            }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::setRenderOrder(RenderOrder order)
{
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            if (!track.accompaniment && track.renderOrder != order)
            {
                if (!changed) pushUndoLocked();
                track.renderOrder = order;
                changed = true;
            }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::setTrackRenderOrder(const juce::String& trackId, RenderOrder order,
                                       std::optional<bool> mergeContiguous)
{
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            if (track.id == trackId && !track.accompaniment
                && (track.renderOrder != order || (mergeContiguous && track.mergeContiguousSource != *mergeContiguous)))
            {
                pushUndoLocked();
                track.renderOrder = order;
                if (mergeContiguous) track.mergeContiguousSource = *mergeContiguous;
                changed = true;
                break;
            }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::moveClip(const juce::String& clipId,double startSeconds)
{
    const auto data=snapshot();for(const auto& track:data.tracks)for(const auto& clip:track.clips)
        if(clip.id==clipId){moveClips({clipId},std::max(0.0,startSeconds)-clip.startSeconds);return;}
}

std::optional<double> ProjectModel::clipSplitPositionLocked(
    const juce::String& clipId, double absoluteSeconds) const
{
    if (!std::isfinite(absoluteSeconds)) return {};
    for (const auto& track : project.tracks)
        for (const auto& clip : track.clips)
            if (clip.id == clipId)
            {
                const auto cut = absoluteSeconds - clip.startSeconds;
                if (cut < 0.01 || cut > clip.durationSeconds - 0.01) return {};
                if (track.accompaniment || !track.compose || clip.notes.empty())
                    return canSplitClipAt(clipId, absoluteSeconds)
                        ? std::optional<double>(absoluteSeconds) : std::nullopt;
                std::vector<std::pair<double, double>> spans;
                spans.reserve(clip.notes.size());
                for (const auto& note : clip.notes)
                {
                    const auto end = note.startSeconds + note.durationSeconds;
                    if (!std::isfinite(note.startSeconds) || !std::isfinite(end)
                        || note.durationSeconds <= 0.0) return {};
                    spans.emplace_back(note.startSeconds, end);
                }
                std::sort(spans.begin(), spans.end());
                // Keep overlapping notes together: the previous note alone
                // cannot tell us whether a longer note still crosses the gap.
                auto reach = spans.front().second;
                std::optional<double> best;
                auto distance = std::numeric_limits<double>::infinity();
                for (std::size_t i = 1; i < spans.size(); ++i)
                {
                    const auto low = std::max(0.01, reach);
                    const auto high = std::min(clip.durationSeconds - 0.01, spans[i].first);
                    if (low <= high + 1.0e-9)
                    {
                        // Treat sub-nanosecond rounding at touching note boundaries as adjacency.
                        const auto candidate = low <= high ? juce::jlimit(low, high, cut) : high;
                        const auto delta = std::abs(candidate - cut);
                        // At equal distance prefer the earlier boundary.
                        if (delta < distance - 1.0e-9)
                        {
                            best = clip.startSeconds + candidate;
                            distance = delta;
                        }
                    }
                    reach = std::max(reach, spans[i].second);
                }
                return best && canSplitClipAt(clipId, *best) ? best : std::nullopt;
            }
    return {};
}

bool ProjectModel::canSplitClip(const juce::String& clipId, double absoluteSeconds) const
{
    const juce::ScopedLock guard(lock);
    return clipSplitPositionLocked(clipId, absoluteSeconds).has_value();
}

juce::String ProjectModel::splitClip(const juce::String& clipId, double absoluteSeconds)
{
    const juce::ScopedLock guard(lock);
    const auto position = clipSplitPositionLocked(clipId, absoluteSeconds);
    return position ? splitClipAt(clipId, *position) : juce::String{};
}

bool ProjectModel::canSplitClipAt(const juce::String& clipId, double absoluteSeconds) const
{
    if (!std::isfinite(absoluteSeconds)) return false;
    const juce::ScopedLock guard(lock);
    for (const auto& track : project.tracks)
        for (const auto& clip : track.clips)
            if (clip.id == clipId)
            {
                const auto cut = absoluteSeconds - clip.startSeconds;
                if (cut < 0.01 || cut > clip.durationSeconds - 0.01) return false;
                for (const auto& note : clip.notes)
                {
                    const auto inside = cut - note.startSeconds;
                    if (inside > 1.0e-8 && inside < note.durationSeconds - 1.0e-8
                        && (inside < 0.01 || note.durationSeconds - inside < 0.01)) return false;
                }
                return true;
            }
    return false;
}

juce::String ProjectModel::splitClipAt(const juce::String& clipId, double absoluteSeconds)
{
    juce::String rightId;
    {
        const juce::ScopedLock guard(lock);
        if (!canSplitClipAt(clipId, absoluteSeconds)) return {};
        for (auto& track : project.tracks)
        {
            const auto found = std::find_if(track.clips.begin(), track.clips.end(),
                [&](const auto& clip) { return clip.id == clipId; });
            if (found == track.clips.end()) continue;
            const auto index = static_cast<std::size_t>(found - track.clips.begin());
            auto original = *found;
            if(trackShowsAllNativeRegions(track))rememberNativeTrimSources(original);
            const auto cut = absoluteSeconds - original.startSeconds;
            const auto audioLength = original.audioLength();
            const auto audioCut = juce::jlimit(0.0, audioLength, cut - original.audioStartSeconds);
            const auto sourceDuration = track.accompaniment ? audioLength
                : (original.sourceDurationSeconds > 1.0e-9 ? original.sourceDurationSeconds : audioLength);
            auto sourceCut = audioLength > 1.0e-9 ? audioCut * sourceDuration / audioLength : 0.0;
            const auto useSourceTimeMap = !track.accompaniment && track.compose
                && !original.notes.empty() && original.sourceTimeMap.size() >= 2;
            if (useSourceTimeMap && audioCut > 0.0 && audioCut < audioLength)
            {
                const auto next = std::upper_bound(original.sourceTimeMap.begin(), original.sourceTimeMap.end(), cut,
                    [](double time, const SourceTimePoint& p) { return time < p.targetSeconds; });
                if (next == original.sourceTimeMap.begin()) sourceCut = next->sourceSeconds;
                else if (next == original.sourceTimeMap.end()) sourceCut = original.sourceTimeMap.back().sourceSeconds;
                else
                {
                    const auto& before = *(next - 1);
                    const auto span = next->targetSeconds - before.targetSeconds;
                    sourceCut = before.sourceSeconds + (next->sourceSeconds - before.sourceSeconds)
                        * (span > 1.0e-9 ? (cut - before.targetSeconds) / span : 0.0);
                }
                sourceCut = juce::jlimit(0.0, sourceDuration, sourceCut);
            }
            pushUndoLocked();
            const juce::ScopedValueSetter<bool> grouping(suppressNestedUndo, true);
            for (const auto& note : original.notes)
            {
                const auto inside = cut - note.startSeconds;
                if (inside > 1.0e-8 && inside < note.durationSeconds - 1.0e-8)
                {
                    const auto newNote = splitNote(note.id, inside);
                    // Classic linear flags are note-local too. The note splitter
                    // already handles DS predictions and offsets separately.
                    for (auto& part : found->notes)
                        if (part.id == newNote)
                            for (auto& curve : part.utauFlagCurves)
                                if (!isDiffSingerParameter(curve.flag))
                                    for (auto& p : curve.points) p.timeSeconds -= inside;
                }
            }
            auto left = *found, right = *found;
            if(trackShowsAllNativeRegions(track))
            {rememberNativeTrimSources(left);right.nativeTrimClock=left.nativeTrimClock;right.nativeSourcePitch=left.nativeSourcePitch;right.nativeSourcePitchComplete=left.nativeSourcePitchComplete;}
            if (!original.parts.empty())
            {
                left.parts = slicedClipParts(original.parts,0,cut,track.accompaniment,track.compose);
                right.parts = slicedClipParts(original.parts,cut,original.durationSeconds,track.accompaniment,track.compose);
            }
            right.id = rightId = makeId("clip");
            left.durationSeconds = cut;
            right.startSeconds = absoluteSeconds;
            right.durationSeconds = original.durationSeconds - cut;
            for(auto& point:right.nativeTrimClock)point.targetSeconds-=cut;
            shiftClipGainEnvelopes(right,-cut);
            anchorClipGainEnvelope(left);anchorClipGainEnvelope(right);
            left.audioStartSeconds = std::min(original.audioStartSeconds, cut);
            left.audioDurationSeconds = audioCut;
            right.audioStartSeconds = std::max(0.0, original.audioStartSeconds - cut);
            right.audioDurationSeconds = audioLength - audioCut;
            left.sourceDurationSeconds = sourceCut;
            right.sourceOffsetSeconds = original.sourceOffsetSeconds + sourceCut;
            right.sourceDurationSeconds = sourceDuration - sourceCut;
            if (!original.parts.empty())
            {
                if (left.parts.empty()) left.audioDurationSeconds = left.sourceDurationSeconds = 0;
                if (right.parts.empty()) right.audioDurationSeconds = right.sourceDurationSeconds = 0;
            }
            left.sourceTimeMap.clear(); right.sourceTimeMap.clear();
            if (useSourceTimeMap)
            {
                if (left.audioDurationSeconds > 0.0)
                    left.sourceTimeMap.push_back({left.audioStartSeconds, 0.0});
                for (const auto& p : original.sourceTimeMap)
                {
                    if (p.targetSeconds > original.audioStartSeconds && p.targetSeconds < cut
                        && p.targetSeconds < original.audioStartSeconds + audioLength) left.sourceTimeMap.push_back(p);
                    if (p.targetSeconds > cut && p.targetSeconds > original.audioStartSeconds
                        && p.targetSeconds < original.audioStartSeconds + audioLength)
                        right.sourceTimeMap.push_back({p.targetSeconds - cut, p.sourceSeconds - sourceCut});
                }
                if (left.audioDurationSeconds > 0.0)
                    left.sourceTimeMap.push_back({left.audioStartSeconds + left.audioDurationSeconds, sourceCut});
                if (right.audioDurationSeconds > 0.0)
                {
                    right.sourceTimeMap.insert(right.sourceTimeMap.begin(), {right.audioStartSeconds, 0.0});
                    right.sourceTimeMap.push_back({right.audioStartSeconds + right.audioDurationSeconds, right.sourceDurationSeconds});
                }
            }
            std::erase_if(left.notes, [&](const auto& note) { return note.startSeconds >= cut - 1.0e-8; });
            std::erase_if(right.notes, [&](const auto& note) { return note.startSeconds < cut - 1.0e-8; });
            for (auto& note : right.notes) note.startSeconds -= cut;
            left.fadeInSeconds = std::min(left.durationSeconds, original.fadeInSeconds);
            right.fadeOutSeconds = std::min(right.durationSeconds, original.fadeOutSeconds);
            left.crossfadeInSeconds = std::min(left.durationSeconds, original.crossfadeInSeconds);
            right.crossfadeOutSeconds = std::min(right.durationSeconds, original.crossfadeOutSeconds);
            left.fadeOutSeconds = audioCut >= audioLength ? original.fadeOutSeconds : 0.0;
            left.crossfadeOutSeconds = audioCut >= audioLength ? original.crossfadeOutSeconds : 0.0;
            right.fadeInSeconds = audioCut <= 0.0 ? original.fadeInSeconds : 0.0;
            right.crossfadeInSeconds = audioCut <= 0.0 ? original.crossfadeInSeconds : 0.0;
            left.glideConnectedToNext = right.glideConnectedFromPrevious = false;
            for(auto* piece:{&left,&right})
                if(piece->parts.size()==1)
                {
                    auto plain=expandedClipParts(*piece,true).front();
                    for(auto& note:plain.notes)note.clipPartId.clear();
                    *piece=std::move(plain);
                }
            track.clips[index] = std::move(left);
            track.clips.insert(track.clips.begin() + static_cast<std::ptrdiff_t>(index + 1), std::move(right));
            break;
        }
    }
    if (rightId.isNotEmpty()) sendChangeMessage();
    return rightId;
}

bool ProjectModel::canMergeClips(const std::vector<juce::String>& clipIds) const
{
    const juce::ScopedLock guard(lock);
    std::vector<juce::String> unique;
    for (const auto& id : clipIds) if (std::find(unique.begin(),unique.end(),id)==unique.end()) unique.push_back(id);
    if (unique.size()<2) return false;
    const TrackData* owner=nullptr;std::size_t found=0;
    for(const auto& track:project.tracks)for(const auto& clip:track.clips)
        if(std::find(unique.begin(),unique.end(),clip.id)!=unique.end())
        { if(owner && owner!=&track)return false;owner=&track;++found; }
    return found==unique.size();
}

juce::String ProjectModel::mergeClips(const std::vector<juce::String>& clipIds)
{
    juce::String mergedId;
    {
        const juce::ScopedLock guard(lock);
        if(!canMergeClips(clipIds))return {};
        for(auto& track:project.tracks)
        {
            std::vector<ClipData> chosen;
            for(const auto& clip:track.clips)
                if(std::find(clipIds.begin(),clipIds.end(),clip.id)!=clipIds.end())chosen.push_back(clip);
            if(chosen.empty())continue;
            std::stable_sort(chosen.begin(),chosen.end(),[](const auto& a,const auto& b){return a.startSeconds<b.startSeconds;});
            ClipData merged;merged.id=mergedId=chosen.front().id;merged.startSeconds=chosen.front().startSeconds;
            merged.sourceFile=chosen.front().sourceFile;merged.durationSeconds=0;
            for(const auto& clip:chosen)
            {
                merged.showNoteHints = merged.showNoteHints || clip.showNoteHints;
                merged.showNormalDisplay = merged.showNormalDisplay || clip.showNormalDisplay;
                merged.durationSeconds=std::max(merged.durationSeconds,clip.startSeconds+clip.durationSeconds-merged.startSeconds);
                for(auto part:expandedClipParts(clip))
                {
                    const auto oldStart=part.startSeconds;
                    part.id=makeId("part");part.startSeconds=oldStart-merged.startSeconds;
                    for(auto note:part.notes)
                    { note.startSeconds+=part.startSeconds;note.clipPartId=part.id;merged.notes.push_back(std::move(note)); }
                    part.notes.clear();merged.parts.push_back(std::move(part));
                }
            }
            std::stable_sort(merged.notes.begin(),merged.notes.end(),[](const auto& a,const auto& b){return a.startSeconds<b.startSeconds;});
            pushUndoLocked();
            const auto insertAt=static_cast<std::size_t>(std::find_if(track.clips.begin(),track.clips.end(),
                [&](const auto& c){return std::find(clipIds.begin(),clipIds.end(),c.id)!=clipIds.end();})-track.clips.begin());
            std::erase_if(track.clips,[&](const auto& c){return std::find(clipIds.begin(),clipIds.end(),c.id)!=clipIds.end();});
            track.clips.insert(track.clips.begin()+static_cast<std::ptrdiff_t>(insertAt),std::move(merged));
            break;
        }
    }
    if(mergedId.isNotEmpty())sendChangeMessage();
    return mergedId;
}

juce::String ProjectModel::duplicateClip(const juce::String& clipId,
                                         double startSeconds,
                                         const juce::String& targetTrackId)
{
    juce::String insertedId;
    {
        const juce::ScopedLock guard(lock);
        auto sourceTrackIndex = project.tracks.size();
        ClipData copy;
        auto found = false;
        for (std::size_t trackIndex = 0; trackIndex < project.tracks.size() && !found;
             ++trackIndex)
            for (const auto& clip : project.tracks[trackIndex].clips)
                if (clip.id == clipId)
                {
                    sourceTrackIndex = trackIndex;
                    copy = clip;
                    found = true;
                    break;
                }
        if (!found || sourceTrackIndex >= project.tracks.size()) return {};

        auto destinationTrackIndex = sourceTrackIndex;
        if (targetTrackId.isNotEmpty())
            for (std::size_t trackIndex = 0; trackIndex < project.tracks.size(); ++trackIndex)
                if (project.tracks[trackIndex].id == targetTrackId)
                {
                    destinationTrackIndex = trackIndex;
                    break;
                }

        pushUndoLocked();
        copy.id = makeId("clip");
        copy.startSeconds = startSeconds >= 0.0
            ? startSeconds : copy.startSeconds + copy.durationSeconds;
        copy.startSeconds = std::max(0.0, copy.startSeconds);
        for (auto& note : copy.notes) note.id = makeId("note");

        // Connection flags at the outer edges describe neighbouring notes in
        // the original timeline.  Preserve joins inside the copied clip, but
        // do not accidentally glide into unrelated material at its new place.
        if (!copy.notes.empty())
        {
            const auto first = std::min_element(copy.notes.begin(), copy.notes.end(),
                [](const auto& left, const auto& right)
                {
                    return left.startSeconds < right.startSeconds;
                });
            const auto last = std::max_element(copy.notes.begin(), copy.notes.end(),
                [](const auto& left, const auto& right)
                {
                    return left.startSeconds + left.durationSeconds
                        < right.startSeconds + right.durationSeconds;
                });
            first->connectedToPrevious = false;
            last->connectedToNext = false;
        }
        auto relative=copy;relative.startSeconds=0;copy.startSeconds=nativePasteStart(project.tracks[destinationTrackIndex],{relative},copy.startSeconds);
        insertedId = copy.id;
        project.tracks[destinationTrackIndex].clips.push_back(std::move(copy));
    }
    sendChangeMessage();
    return insertedId;
}

void ProjectModel::resizeClip(const juce::String& clipId, double startSeconds,
                              double durationSeconds)
{
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
                if (clip.id == clipId)
                {
                    const auto oldDuration = std::max(0.01, clip.durationSeconds);
                    auto nextDuration = std::max(0.01, durationSeconds);
                    auto nextStart = std::max(0.0, startSeconds);
                    constrainNativeClipResize(track,clipId,nextStart,nextDuration);
                    if (std::abs(clip.startSeconds - nextStart) <= 1.0e-9
                        && std::abs(oldDuration - nextDuration) <= 1.0e-9) return;
                    pushUndoLocked();
                    const auto ratio = nextDuration / oldDuration;
                    const auto native = trackShowsAllNativeRegions(track);
                    if(native && std::abs(ratio-1.0)>1.e-9)
                    {
                        // Freeze the source clock before changing consonant
                        // times/velocity. Its UI velocity range cannot express
                        // arbitrary long stretches without moving source anchors.
                        if(clip.parts.empty())clip.sourceTimeMap=nativeClipClock(clip);
                        else for(const auto& source:expandedClipParts(clip))
                            for(auto& part:clip.parts)if(source.id==clip.id+":"+part.id)
                            {
                                part.sourceTimeMap=nativeClipClock(source);
                                for(auto& point:part.sourceTimeMap)
                                    point.targetSeconds+=source.startSeconds-clip.startSeconds-part.startSeconds;
                            }
                        rememberNativeTrimSources(clip);
                    }
                    const auto scaleGains=[&](ClipData& c)
                    {
                        if(!native)return;
                        for(auto& p:c.gainEnvelope)p.timeSeconds*=ratio;
                        for(auto& layer:c.inheritedGainEnvelopes)for(auto& p:layer)p.timeSeconds*=ratio;
                    };
                    scaleGains(clip);
                    clip.audioStartSeconds *= ratio;
                    if (clip.audioDurationSeconds >= 0.0) clip.audioDurationSeconds *= ratio;
                    for(auto& part:clip.parts)
                    {
                        part.startSeconds*=ratio;part.durationSeconds*=ratio;part.audioStartSeconds*=ratio;
                        if(part.audioDurationSeconds>=0)part.audioDurationSeconds*=ratio;
                        for(auto& point:part.sourceTimeMap)point.targetSeconds*=ratio;
                        for(auto& point:part.nativeTrimClock)point.targetSeconds*=ratio;
                        scaleGains(part);
                        part.fadeInSeconds*=ratio;part.fadeOutSeconds*=ratio;
                        part.crossfadeInSeconds*=ratio;part.crossfadeOutSeconds*=ratio;
                    }
                    for (auto& note : clip.notes)
                    {
                        note.startSeconds *= ratio;
                        note.durationSeconds *= ratio;
                        note.consonantSeconds *= ratio;
                        // attackSpeed is the source-time / element-time slope.
                        // Keep the source Attack boundary fixed while its target
                        // position stretches with the rest of the clip.
                        note.attackSpeed = juce::jlimit(0.05f, 20.0f,
                            note.attackSpeed / static_cast<float>(ratio));
                        for (auto& point : note.contour) point.timeSeconds *= ratio;
                        for (auto& point : note.pitchControlPoints) point.timeSeconds *= ratio;
                        for (auto& point : note.diffSingerPitchReference) point.timeSeconds *= ratio;
                        for (auto& point : note.diffSingerPitchOffset) point.timeSeconds *= ratio;
                        if(native)
                        {
                            for(auto& point:note.amplitudeEnvelope)point.timeSeconds*=ratio;
                            note.vibratoReferenceDurationSeconds*=ratio;
                            note.vibratoTimeOffsetSeconds*=ratio;
                        }
                        mapDiffSingerParameterTimes(note,[&](double time){return time*ratio;});
                        for (auto& marker : note.sibilantMarkers) marker *= ratio;
                    }
                    // The selected source range is unchanged by a timeline
                    // stretch.  Move only the target side of the imported warp
                    // so the original Melodyne Attack/vowel source anchors are
                    // retained exactly.
                    for (auto& point : clip.sourceTimeMap)
                        point.targetSeconds *= ratio;
                    for (auto& point : clip.nativeTrimClock) point.targetSeconds *= ratio;
                    clip.fadeInSeconds = std::min(nextDuration, clip.fadeInSeconds * ratio);
                    clip.fadeOutSeconds = std::min(nextDuration, clip.fadeOutSeconds * ratio);
                    clip.crossfadeInSeconds = std::min(nextDuration, clip.crossfadeInSeconds * ratio);
                    clip.crossfadeOutSeconds = std::min(nextDuration, clip.crossfadeOutSeconds * ratio);
                    clip.startSeconds = nextStart;
                    clip.durationSeconds = nextDuration;
                    changed = true;
                    break;
                }
    }
    if (changed) sendChangeMessage();
}

bool ProjectModel::setClipEditableLength(const juce::String& clipId, double durationSeconds)
{
    if (!std::isfinite(durationSeconds) || durationSeconds < 0.01) return false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
                if (clip.id == clipId)
                {
                    if (!std::isfinite(clip.startSeconds + durationSeconds)) return false;
                    if (std::abs(clip.durationSeconds - durationSeconds) < 1.e-9) return true;
                    pushUndoLocked();
                    // Older audio regions use the outer length as their audio clock.
                    // Make that clock explicit before editing only the outer boundary.
                    if (clip.audioDurationSeconds < 0.0 && (clip.sourceFile != juce::File{} || !clip.parts.empty()))
                        clip.audioDurationSeconds = clip.durationSeconds;
                    clip.durationSeconds = durationSeconds;
                    sendChangeMessage();
                    return true;
                }
    }
    return false;
}

bool ProjectModel::trimClip(const juce::String& clipId, double startSeconds, double durationSeconds)
{
    if (!std::isfinite(startSeconds) || !std::isfinite(durationSeconds)
        || startSeconds < 0.0 || durationSeconds < 0.01) return false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
        {
            const auto find = [&]() { return std::find_if(track.clips.begin(), track.clips.end(),
                [&](const auto& c) { return c.id == clipId; }); };
            auto found = find();
            if (found == track.clips.end()) continue;
            const auto oldStart = found->startSeconds;
            const auto oldEnd = oldStart + found->durationSeconds;
            const auto end = startSeconds + durationSeconds;
            if (std::abs(startSeconds - oldStart) < 1.0e-9 && std::abs(end - oldEnd) < 1.0e-9) return false;
            const auto overlaps = std::min(end, oldEnd) - std::max(startSeconds, oldStart) >= 0.01;
            if (overlaps && ((startSeconds > oldStart && !canSplitClipAt(clipId, startSeconds))
                || (end < oldEnd && !canSplitClipAt(clipId, end)))) return false;
            pushUndoLocked();
            const juce::ScopedValueSetter<bool> grouping(suppressNestedUndo, true);
            if (overlaps)
            {
                if (startSeconds > oldStart)
                {
                    const auto right = splitClipAt(clipId, startSeconds);
                    std::erase_if(track.clips, [&](const auto& c) { return c.id == clipId; });
                    for (auto& c : track.clips) if (c.id == right) c.id = clipId;
                }
                if (end < oldEnd)
                {
                    const auto right = splitClipAt(clipId, end);
                    std::erase_if(track.clips, [&](const auto& c) { return c.id == right; });
                }
                found = find();
                const auto shift = found->startSeconds - startSeconds;
                found->audioDurationSeconds = found->audioLength();
                found->audioStartSeconds += shift;
                shiftClipGainEnvelopes(*found,shift);
                for(auto& part:found->parts)part.startSeconds+=shift;
                for (auto& note : found->notes) note.startSeconds += shift;
                for (auto& point : found->sourceTimeMap) point.targetSeconds += shift;
                for (auto& point : found->nativeTrimClock) point.targetSeconds += shift;
            }
            else
            {
                found->notes.clear(); found->sourceTimeMap.clear();
                found->parts.clear();
                found->gainEnvelope.clear();found->inheritedGainEnvelopes.clear();
                found->audioStartSeconds = 0.0; found->audioDurationSeconds = 0.0;
                found->sourceDurationSeconds = 0.0;
                found->fadeInSeconds = found->fadeOutSeconds = 0.0;
                found->crossfadeInSeconds = found->crossfadeOutSeconds = 0.0;
            }
            found->startSeconds = startSeconds;
            found->durationSeconds = durationSeconds;
            anchorClipGainEnvelope(*found);
            found->glideConnectedFromPrevious = found->glideConnectedToNext = false;
            sendChangeMessage();
            return true;
        }
    }
    return false;
}

void ProjectModel::setClipGain(const juce::String& clipId, float gain)
{
    const auto next = juce::jlimit(0.0f, 4.0f, gain);
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
                if (clip.id == clipId)
                {
                    if (clip.gain == next) return;
                    pushUndoLocked();
                    clip.gain = next;
                    changed = true;
                    break;
                }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::setClipFades(const juce::String& clipId, double fadeInSeconds,
                                double fadeOutSeconds)
{
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
                if (clip.id == clipId)
                {
                    const auto nextIn = juce::jlimit(0.0, clip.durationSeconds,
                                                     fadeInSeconds);
                    const auto nextOut = juce::jlimit(0.0, clip.durationSeconds,
                                                      fadeOutSeconds);
                    if (std::abs(clip.fadeInSeconds - nextIn) <= 1.0e-9
                        && std::abs(clip.fadeOutSeconds - nextOut) <= 1.0e-9) return;
                    pushUndoLocked();
                    clip.fadeInSeconds = nextIn;
                    clip.fadeOutSeconds = nextOut;
                    changed = true;
                    break;
                }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::setClipMuted(const juce::String& clipId, bool muted)
{
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
                if (clip.id == clipId)
                {
                    if (clip.muted == muted) return;
                    pushUndoLocked();
                    clip.muted = muted;
                    changed = true;
                    break;
                }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::setClipNoteHints(const juce::String& clipId, bool enabled)
{
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
                if (clip.id == clipId)
                {
                    if (clip.showNoteHints == enabled || (enabled
                        && (track.accompaniment || !track.compose || clip.notes.empty()))) return;
                    pushUndoLocked();
                    clip.showNoteHints = enabled;
                    changed = true;
                    break;
                }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::setClipNormalDisplay(const juce::String& clipId, bool enabled)
{
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
                if (clip.id == clipId)
                {
                    if (clip.showNormalDisplay == enabled || (enabled
                        && (track.accompaniment || !track.compose))) return;
                    pushUndoLocked();
                    clip.showNormalDisplay = enabled;
                    changed = true;
                    break;
                }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::removeClip(const juce::String& clipId)
{
    removeClips({clipId});
}

void ProjectModel::moveClips(const std::vector<juce::String>& clipIds, double deltaSeconds, int trackDelta)
{
    if (!std::isfinite(deltaSeconds)) return;
    {
        const juce::ScopedLock guard(lock);
        double earliest = std::numeric_limits<double>::max();
        auto firstTrack = static_cast<int>(project.tracks.size()), lastTrack = -1;
        const auto selected = [&](const ClipData& clip) {
            return std::find(clipIds.begin(), clipIds.end(), clip.id) != clipIds.end();
        };
        for (int row = 0; row < static_cast<int>(project.tracks.size()); ++row)
            for (const auto& clip : project.tracks[static_cast<std::size_t>(row)].clips)
                if (selected(clip)) {
                    earliest = std::min(earliest, clip.startSeconds);
                    firstTrack = std::min(firstTrack, row); lastTrack = std::max(lastTrack, row);
                }
        if (lastTrack < 0) return;
        deltaSeconds = std::max(-earliest, deltaSeconds);
        trackDelta = juce::jlimit(-firstTrack, static_cast<int>(project.tracks.size())-1-lastTrack, trackDelta);
        deltaSeconds=constrainedNativeClipMoves(project,clipIds,deltaSeconds,trackDelta);
        if (std::abs(deltaSeconds) < 1.0e-9 && trackDelta == 0) return;
        pushUndoLocked();
        if (trackDelta == 0) {
            for (auto& track : project.tracks) for (auto& clip : track.clips)
                if (selected(clip)) clip.startSeconds += deltaSeconds;
        } else {
            // Remove the entire selection before inserting anything. Otherwise
            // a clip transferred downwards could be moved a second time.
            std::vector<std::pair<std::size_t, ClipData>> moved;
            for (int row = 0; row < static_cast<int>(project.tracks.size()); ++row) {
                auto& clips = project.tracks[static_cast<std::size_t>(row)].clips;
                for (auto it = clips.begin(); it != clips.end();) {
                    if (!selected(*it)) { ++it; continue; }
                    it->startSeconds += deltaSeconds;
                    moved.emplace_back(static_cast<std::size_t>(row+trackDelta), std::move(*it));
                    it = clips.erase(it);
                }
            }
            for (auto& [row, clip] : moved) project.tracks[row].clips.push_back(std::move(clip));
            for (auto& track : project.tracks)
                std::stable_sort(track.clips.begin(), track.clips.end(), [](const auto& a, const auto& b) {
                    return a.startSeconds < b.startSeconds;
                });
        }
    }
    sendChangeMessage();
}

void ProjectModel::removeClips(const std::vector<juce::String>& clipIds)
{
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
        {
            const auto selected = [&](const auto& clip)
            { return std::find(clipIds.begin(), clipIds.end(), clip.id) != clipIds.end(); };
            if (!std::any_of(track.clips.begin(), track.clips.end(), selected)) continue;
            if (!changed) pushUndoLocked();
            std::erase_if(track.clips, selected);
            changed = true;
        }
    }
    if (changed) sendChangeMessage();
}

bool ProjectModel::moveTrackBefore(const juce::String& trackId, const juce::String& beforeTrackId)
{
    {
        const juce::ScopedLock guard(lock);
        auto& tracks = project.tracks;
        const auto source = std::find_if(tracks.begin(), tracks.end(), [&](const auto& t){return t.id == trackId;});
        const auto before = std::find_if(tracks.begin(), tracks.end(), [&](const auto& t){return t.id == beforeTrackId;});
        if (source == tracks.end() || (beforeTrackId.isNotEmpty() && before == tracks.end()) || source == before) return false;
        const auto from = std::distance(tracks.begin(), source);
        auto to = std::distance(tracks.begin(), before);
        if (to > from) --to;
        if (from == to) return false;
        pushUndoLocked();
        if (from < to) std::rotate(tracks.begin()+from, tracks.begin()+from+1, tracks.begin()+to+1);
        else std::rotate(tracks.begin()+to, tracks.begin()+from, tracks.begin()+from+1);
    }
    sendChangeMessage();
    return true;
}

void ProjectModel::removeTrack(const juce::String& trackId)
{
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        const auto found = std::find_if(project.tracks.begin(), project.tracks.end(),
            [&](const auto& track) { return track.id == trackId; });
        if (found != project.tracks.end())
        {
            pushUndoLocked();
            project.tracks.erase(found);
            changed = true;
        }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::transposeNote(const juce::String& noteId, float semitones)
{
    transposeNotes({ noteId }, semitones);
}

void ProjectModel::transposeNotes(const std::vector<juce::String>& noteIds, float semitones)
{
    const auto includes = [&](const juce::String& id)
    {
        return std::find(noteIds.begin(), noteIds.end(), id) != noteIds.end();
    };
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (const auto& track : project.tracks)
            for (const auto& clip : track.clips)
                for (const auto& note : clip.notes)
                    changed = changed || includes(note.id);
        if (!changed) return;
        pushUndoLocked();
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
                for (auto& note : clip.notes)
                    if (includes(note.id))
                    {
                        const auto previous = note.midiNote;
                        note.midiNote = juce::jlimit(0.0f, 127.0f, note.midiNote + semitones);
                        const auto applied = note.midiNote - previous;
                        for (auto& point : note.pitchControlPoints)
                            point.targetMidi = juce::jlimit(0.0f, 127.0f,
                                point.targetMidi + applied);
                        for (auto& point : note.diffSingerPitchReference)
                            point.targetMidi = juce::jlimit(0.0f, 127.0f,
                                point.targetMidi + applied);
                    }
    }
    if (changed) sendChangeMessage();
}

bool ProjectModel::trimNativeNoteEdge(const juce::String& id, double delta, bool left)
{
    if (!std::isfinite(delta) || std::abs(delta) < 1.e-9) return false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks) if (trackShowsAllNativeRegions(track))
            for (auto& clip : track.clips)
                if (auto plan = planNativeNoteTrim(clip, id, delta, left, track.allowNativeAudioOverlap))
                {
                    const auto safe=constrainedNativeTrimDelta(track,id,plan->delta,left);
                    plan=planNativeNoteTrim(clip,id,safe,left,track.allowNativeAudioOverlap);
                    if(!plan)return false;
                    if (std::abs(plan->delta) < 1.e-9) return false;
                    pushUndoLocked(); clip = std::move(plan->clip);
                    const auto cut = [&](const auto& link) { return left ? link.rightNoteId == id : link.leftNoteId == id; };
                    for (auto& t : project.tracks) for (auto& c : t.clips) for (auto& n : c.notes)
                        for (const auto& link : project.nativeConnections) if (cut(link))
                        {
                            if (n.id == link.leftNoteId) n.connectedToNext = false;
                            if (n.id == link.rightNoteId) n.connectedToPrevious = false;
                        }
                    std::erase_if(project.nativeConnections, cut);
                    sendChangeMessage(); return true;
                }
    }
    return false;
}

bool ProjectModel::resizeNativeNoteEdge(const juce::String& id, double delta, bool left)
{
    if (!std::isfinite(delta) || std::abs(delta)<1.e-9) return false;
    bool changed=false;
    {
        const juce::ScopedLock guard(lock);
        for(auto& track:project.tracks) if(trackShowsAllNativeRegions(track))
            for(auto& clip:track.clips)
                if(auto plan=planNativeNoteMove(clip,{id},delta,0,
                    left?NativeNoteTimeEdit::leftEdge:NativeNoteTimeEdit::rightEdge))
                {
                    const auto safe=constrainedNativeNoteDelta(track,{id},plan->delta,
                        left?NativeNoteTimeEdit::leftEdge:NativeNoteTimeEdit::rightEdge);
                    plan=planNativeNoteMove(clip,{id},safe,0,left?NativeNoteTimeEdit::leftEdge:NativeNoteTimeEdit::rightEdge);
                    if(!plan||std::abs(plan->delta)<1.e-9)return false;
                    pushUndoLocked();clip=std::move(plan->clip);changed=true;
                    break;
                }
    }
    if(changed) sendChangeMessage();
    return changed;
}

bool ProjectModel::alignNotePreutteranceToGrid(const juce::String& noteId, double quarterStep, juce::String& error)
{
    error.clear();
    if (!std::isfinite(quarterStep) || quarterStep < 1.e-6)
    { error = "Invalid musical grid step"; return false; }
    bool changed = false;
    {
        const juce::ScopedLock guard(lock);
        TrackData* owner = nullptr; ClipData* clip = nullptr; const NoteData* note = nullptr;
        for (auto& track : project.tracks) for (auto& candidate : track.clips)
            for (const auto& value : candidate.notes) if (value.id == noteId)
            { owner = &track; clip = &candidate; note = &value; }
        if (!owner || !clip || !note) { error = "Unknown note"; return false; }
        bool physical = clip->sourceFile != juce::File{};
        for (const auto& part : clip->parts) if (part.id == note->clipPartId && part.sourceFile != juce::File{}) physical = true;
        const auto originalNote = *note;
        const auto pre = clip->startSeconds + (physical ? nativeNotePreutteranceTarget(*note, *clip) : note->startSeconds);
        double from = clip->startSeconds;
        double to = std::max(project.durationSeconds() + note->durationSeconds + 1.0,
            project.secondsForQuarterPosition(project.quarterPositionForSeconds(pre) + quarterStep * 2));
        if (physical)
        {
            const auto low = planNativeNoteMove(*clip, {noteId}, -1.e6, 0, NativeNoteTimeEdit::move);
            const auto high = planNativeNoteMove(*clip, {noteId}, 1.e6, 0, NativeNoteTimeEdit::move);
            if (!low || !high) { error = "No elastic vowel is available for alignment"; return false; }
            from = std::max(0.0, pre + low->delta); to = std::min(to, pre + high->delta);
        }
        const auto first = static_cast<juce::int64>(std::ceil(project.quarterPositionForSeconds(from) / quarterStep - 1.e-9));
        const auto last = static_cast<juce::int64>(std::floor(project.quarterPositionForSeconds(to) / quarterStep + 1.e-9));
        if (last < first || last - first > 200000)
        { error = "No feasible grid point in the timing range"; return false; }
        std::vector<double> targets;
        for (auto index = first; index <= last; ++index) targets.push_back(project.secondsForQuarterPosition(index * quarterStep));
        std::stable_sort(targets.begin(), targets.end(), [pre](double a, double b) { return std::abs(a - pre) < std::abs(b - pre); });
        for (const auto target : targets)
        {
            const auto delta = target - pre;
            if (std::abs(delta) < 1.e-8) return false; // already aligned, no undo
            if (physical)
            {
                auto plan = planNativeNoteMove(*clip, {noteId}, delta, 0, NativeNoteTimeEdit::move);
                if (!plan || std::abs(plan->delta - delta) > 1.e-7) continue;
                const auto safe = constrainedNativeNoteDelta(*owner, {noteId}, delta, NativeNoteTimeEdit::move);
                if (std::abs(safe - delta) > 1.e-7) continue; // never clamp off-grid
                auto moved = std::find_if(plan->clip.notes.begin(), plan->clip.notes.end(), [&](const auto& value) { return value.id == noteId; });
                if (moved == plan->clip.notes.end()) continue;
                // Alignment translates the complete selected pronunciation.
                // Its local controls (including out-of-body shared handles and
                // vibrato phase) do not undergo an implicit resize.
                const auto movedStart = moved->startSeconds;
                *moved = originalNote; moved->startSeconds = movedStart;
                if (std::abs(plan->clip.startSeconds + nativeNotePreutteranceTarget(*moved, plan->clip) - target) > 1.e-7) continue;
                pushUndoLocked(); *clip = std::move(plan->clip); changed = true; break;
            }
            // Reuse score movement rather than introducing another insertion or
            // collision algorithm. Only the final candidate enters live history.
            ProjectModel draft; draft.resetDocument(project); juce::String moveError;
            if (owner->pitchAlgorithm == PitchAlgorithm::utau)
            {
                if (!draft.moveUtauNotes({noteId}, delta, 0, &moveError)) continue;
            }
            else draft.resizeNote(noteId, originalNote.startSeconds + delta, originalNote.durationSeconds);
            auto proposed = draft.snapshot(); bool exact = false;
            for (auto& track : proposed.tracks) for (auto& candidate : track.clips) for (auto& value : candidate.notes)
                if (value.id == noteId)
                {
                    exact = std::abs(candidate.startSeconds + value.startSeconds - target) < 1.e-7;
                    if (owner->pitchAlgorithm != PitchAlgorithm::utau)
                    {
                        const auto start = value.startSeconds;
                        value = originalNote; value.startSeconds = start; // rigid move: retain all placed curve data
                    }
                }
            if (!exact) continue;
            pushUndoLocked(); project = std::move(proposed); changed = true; break;
        }
        if (!changed) error = "No feasible grid point: protected consonant/preutterance or neighbouring source cannot be compressed";
    }
    if (changed) sendChangeMessage();
    return changed;
}

bool ProjectModel::moveNativeNotes(const std::vector<juce::String>& ids,
                                   double seconds, float semitones, juce::String* error)
{
    if (error) error->clear();
    if (ids.empty() || !std::isfinite(seconds) || !std::isfinite(semitones)) return false;
    bool changed = false;
    {
        const juce::ScopedLock guard(lock);
        std::vector<ClipData*> owners;
        std::size_t count = 0;
        for (auto& track : project.tracks) for (auto& clip : track.clips)
        {
            const auto found = std::count_if(clip.notes.begin(), clip.notes.end(), [&](const auto& note)
                { return std::find(ids.begin(), ids.end(), note.id) != ids.end(); });
            if (!found) continue;
            if (!trackShowsAllNativeRegions(track)) return false;
            owners.push_back(&clip); count += found;
        }
        if (count != ids.size()) return false;
        auto delta = seconds; float low = -127, high = 127;
        for (const auto* clip : owners)
        {
            for (const auto& note : clip->notes) if (std::find(ids.begin(),ids.end(),note.id)!=ids.end())
            { low=std::max(low,-note.midiNote); high=std::min(high,127-note.midiNote); }
            const auto plan = planNativeNoteMove(*clip, ids, seconds, 0);
            if (!plan) return false;
            delta = seconds < 0 ? std::max(delta,plan->delta) : std::min(delta,plan->delta);
        }
        for(const auto& track:project.tracks)
        {const auto safe=constrainedNativeNoteDelta(track,ids,delta);delta=delta<0?std::max(delta,safe):std::min(delta,safe);}
        const auto pitch = juce::jlimit(low,high,semitones);
        if (std::abs(delta)<1.e-9 && std::abs(pitch)<1.e-6) return false;
        std::vector<NativeNoteMovePlan> plans;
        for (const auto* clip : owners)
        {
            auto plan = planNativeNoteMove(*clip,ids,delta,pitch);
            if (!plan) return false;
            plans.push_back(std::move(*plan));
        }
        pushUndoLocked();
        for (std::size_t i=0;i<owners.size();++i) *owners[i]=std::move(plans[i].clip);
        changed=true;
    }
    if (changed) sendChangeMessage();
    return changed;
}

bool ProjectModel::moveUtauNotes(const std::vector<juce::String>& noteIds,
                                 double deltaSeconds, float semitones, juce::String* error)
{
    if (error) error->clear();
    if (!std::isfinite(deltaSeconds) || !std::isfinite(semitones)) return false;
    if (noteIds.empty()) return false;
    const auto includes = [&](const juce::String& id)
    {
        return std::find(noteIds.begin(), noteIds.end(), id) != noteIds.end();
    };

    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        TrackData* ownerTrack = nullptr;
        ClipData* ownerClip = nullptr;
        std::vector<ClipData*> ownerClips;
        std::size_t foundCount = 0;
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
            {
                const auto count = static_cast<std::size_t>(std::count_if(
                    clip.notes.begin(), clip.notes.end(),
                    [&](const auto& note) { return includes(note.id); }));
                if (count == 0) continue;
                if (ownerTrack != nullptr && ownerTrack != &track) return false;
                ownerTrack = &track;
                ownerClip = &clip;
                foundCount += count;
                ownerClips.push_back(&clip);
            }
        if (ownerTrack == nullptr || ownerClip == nullptr
            || ownerTrack->pitchAlgorithm != PitchAlgorithm::utau
            || foundCount != noteIds.size())
            return false;

        if (ownerClips.size()>1)
        {
            auto time=deltaSeconds; float low=-127,high=127;
            for(const auto* clip:ownerClips) for(const auto& note:clip->notes) if(includes(note.id))
            {time=std::max(time,-note.startSeconds);low=std::max(low,-note.midiNote);high=std::min(high,127-note.midiNote);}
            const auto pitch=juce::jlimit(low,high,semitones);
            if(std::abs(time)<1.e-9 && std::abs(pitch)<1.e-6f)return false;
            // A group spanning regions must stay rigid. Do not independently
            // insert/ripple one region, which would silently change the rhythm.
            if(std::abs(time)>1.e-9)
                for(const auto* clip:ownerClips) for(const auto& note:clip->notes) if(includes(note.id))
                    for(const auto& other:clip->notes) if(!includes(other.id)
                        && note.startSeconds+time<other.startSeconds+other.durationSeconds-1.e-9
                        && note.startSeconds+time+note.durationSeconds>other.startSeconds+1.e-9)
                    {
                        if(error)*error=juce::String::fromUTF8("无法整体移动：目标位置与未选音符重叠。选区和音符均已保留。");
                        return false;
                    }
            pushUndoLocked();
            for(auto* clip:ownerClips)
            {
                for(auto& note:clip->notes) if(includes(note.id))
                {
                    note.startSeconds+=time;note.midiNote+=pitch;
                    for(auto& point:note.pitchControlPoints)point.targetMidi=juce::jlimit(0.f,127.f,point.targetMidi+pitch);
                    for(auto& point:note.diffSingerPitchReference)point.targetMidi=juce::jlimit(0.f,127.f,point.targetMidi+pitch);
                    clip->durationSeconds=std::max(clip->durationSeconds,note.startSeconds+note.durationSeconds);
                }
                std::stable_sort(clip->notes.begin(),clip->notes.end(),[](const auto& a,const auto& b){return a.startSeconds<b.startSeconds;});
            }
            sendChangeMessage();return true;
        }

        auto phraseStart = std::numeric_limits<double>::max();
        auto phraseEnd = 0.0;
        auto minimumPitchDelta = -127.0f;
        auto maximumPitchDelta = 127.0f;
        for (const auto& note : ownerClip->notes)
            if (includes(note.id))
            {
                phraseStart = std::min(phraseStart, note.startSeconds);
                phraseEnd = std::max(phraseEnd,
                    note.startSeconds + note.durationSeconds);
                minimumPitchDelta = std::max(minimumPitchDelta, -note.midiNote);
                maximumPitchDelta = std::min(maximumPitchDelta, 127.0f - note.midiNote);
            }
        if (phraseStart == std::numeric_limits<double>::max()) return false;

        const auto phraseDuration = std::max(0.01, phraseEnd - phraseStart);
        auto destination = std::max(0.0, phraseStart + deltaSeconds);
        const auto appliedPitch = juce::jlimit(
            minimumPitchDelta, maximumPitchDelta, semitones);

        // Treat the selected notes as one phrase.  UTAU is monophonic, so a
        // time collision is meaningful regardless of the displayed pitch row.
        auto collisionStart = std::numeric_limits<double>::max();
        const auto destinationEnd = destination + phraseDuration;
        for (const auto& note : ownerClip->notes)
            if (!includes(note.id)
                && note.startSeconds < destinationEnd - 1.0e-9
                && note.startSeconds + note.durationSeconds > destination + 1.0e-9)
                collisionStart = std::min(collisionStart, note.startSeconds);

        const auto insertsAtCollision =
            collisionStart != std::numeric_limits<double>::max();
        if (insertsAtCollision)
        {
            // A move is a cut followed by an insert, not a copy followed by
            // an insert.  When the target lies to the right, removing the
            // phrase first moves that target left by exactly phraseDuration.
            destination = collisionStart >= phraseEnd - 1.0e-9
                ? collisionStart - phraseDuration : collisionStart;
        }
        const auto appliedTime = destination - phraseStart;
        changed = std::abs(appliedTime) > 1.0e-9
            || std::abs(appliedPitch) > 1.0e-6f || insertsAtCollision;
        if (!changed) return false;

        pushUndoLocked();
        if (insertsAtCollision)
        {
            // Close the source slot first, then open an equal-sized slot at
            // the destination.  Notes after both locations therefore keep
            // their original absolute positions and no extra gap accumulates.
            for (auto& note : ownerClip->notes)
                if (!includes(note.id)
                    && note.startSeconds >= phraseEnd - 1.0e-9)
                    note.startSeconds -= phraseDuration;
            for (auto& note : ownerClip->notes)
                if (!includes(note.id)
                    && note.startSeconds >= destination - 1.0e-9)
                    note.startSeconds += phraseDuration;
        }

        for (auto& note : ownerClip->notes)
            if (includes(note.id))
            {
                note.startSeconds += appliedTime;
                note.midiNote = juce::jlimit(0.0f, 127.0f,
                                             note.midiNote + appliedPitch);
                for (auto& point : note.pitchControlPoints)
                    point.targetMidi = juce::jlimit(0.0f, 127.0f,
                        point.targetMidi + appliedPitch);
                for (auto& point : note.diffSingerPitchReference)
                    point.targetMidi = juce::jlimit(0.0f, 127.0f,
                        point.targetMidi + appliedPitch);
            }

        std::stable_sort(ownerClip->notes.begin(), ownerClip->notes.end(),
            [](const auto& left, const auto& right)
            {
                return left.startSeconds < right.startSeconds;
            });
        for (const auto& note : ownerClip->notes)
            ownerClip->durationSeconds = std::max(ownerClip->durationSeconds,
                note.startSeconds + note.durationSeconds);
    }
    if (changed) sendChangeMessage();
    return changed;
}

void ProjectModel::setNotesMidi(const std::vector<juce::String>& noteIds, float midiNote)
{
    const auto target = juce::jlimit(0.0f, 127.0f, midiNote);
    const auto includes = [&](const juce::String& id)
    {
        return std::find(noteIds.begin(), noteIds.end(), id) != noteIds.end();
    };
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (const auto& track : project.tracks)
            for (const auto& clip : track.clips)
                for (const auto& note : clip.notes)
                    changed = changed || (includes(note.id)
                        && std::abs(note.midiNote - target) > 1.0e-6f);
        if (!changed) return;
        pushUndoLocked();
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
                for (auto& note : clip.notes)
                    if (includes(note.id))
                    {
                        const auto applied = target - note.midiNote;
                        note.midiNote = target;
                        for (auto& point : note.pitchControlPoints)
                            point.targetMidi = juce::jlimit(0.0f, 127.0f,
                                point.targetMidi + applied);
                        for (auto& point : note.diffSingerPitchReference)
                            point.targetMidi = juce::jlimit(0.0f, 127.0f,
                                point.targetMidi + applied);
                    }
    }
    sendChangeMessage();
}

void ProjectModel::averageNotesMidi(const std::vector<juce::String>& noteIds)
{
    const auto includes = [&](const juce::String& id)
    {
        return std::find(noteIds.begin(), noteIds.end(), id) != noteIds.end();
    };
    auto sum = 0.0;
    auto count = 0;
    {
        const juce::ScopedLock guard(lock);
        for (const auto& track : project.tracks)
            for (const auto& clip : track.clips)
                for (const auto& note : clip.notes)
                    if (includes(note.id))
                    {
                        sum += note.midiNote;
                        ++count;
                    }
    }
    if (count > 0) setNotesMidi(noteIds, static_cast<float>(sum / count));
}

void ProjectModel::quantizeNotesMidi(const std::vector<juce::String>& noteIds,
                                     float stepSemitones)
{
    const auto step = juce::jlimit(0.01f, 12.0f, stepSemitones);
    const auto includes = [&](const juce::String& id)
    {
        return std::find(noteIds.begin(), noteIds.end(), id) != noteIds.end();
    };
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (const auto& track : project.tracks)
            for (const auto& clip : track.clips)
                for (const auto& note : clip.notes)
                    if (includes(note.id))
                    {
                        const auto target = juce::jlimit(0.0f, 127.0f,
                            std::round(note.midiNote / step) * step);
                        changed = changed || std::abs(note.midiNote - target) > 1.0e-6f;
                    }
        if (!changed) return;
        pushUndoLocked();
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
                for (auto& note : clip.notes)
                    if (includes(note.id))
                    {
                        const auto target = juce::jlimit(0.0f, 127.0f,
                            std::round(note.midiNote / step) * step);
                        const auto applied = target - note.midiNote;
                        note.midiNote = target;
                        for (auto& point : note.pitchControlPoints)
                            point.targetMidi = juce::jlimit(0.0f, 127.0f,
                                point.targetMidi + applied);
                        for (auto& point : note.diffSingerPitchReference)
                            point.targetMidi = juce::jlimit(0.0f, 127.0f,
                                point.targetMidi + applied);
                    }
    }
    sendChangeMessage();
}

void ProjectModel::removeNotesRippling(const std::vector<juce::String>& noteIds)
{
    if (noteIds.empty()) return;
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
            {
                // The stretch the chosen notes take up, from the first start
                // to the last end.  What follows closes up against what came
                // before, whether either of those is a note or a silence.
                auto from = std::numeric_limits<double>::max();
                auto to = -std::numeric_limits<double>::max();
                for (const auto& note : clip.notes)
                    if (std::find(noteIds.begin(), noteIds.end(), note.id) != noteIds.end())
                    {
                        from = std::min(from, note.startSeconds);
                        to = std::max(to, note.startSeconds + note.durationSeconds);
                    }
                if (to <= from) continue;
                const auto span = to - from;
                if (!changed) pushUndoLocked();
                changed = true;
                std::erase_if(clip.notes, [&noteIds](const auto& note)
                {
                    return std::find(noteIds.begin(), noteIds.end(), note.id)
                        != noteIds.end();
                });
                for (auto& note : clip.notes)
                    if (note.startSeconds >= to - 1.0e-9)
                        note.startSeconds = std::max(0.0, note.startSeconds - span);
                auto needed = 0.0;
                for (const auto& note : clip.notes)
                    needed = std::max(needed, note.startSeconds + note.durationSeconds);
                clip.durationSeconds = std::max(0.01,
                    std::max(needed, clip.durationSeconds - span));
            }
        dropEmptyRecordedClipsLocked();
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::insertGapBeforeNote(const juce::String& noteId, double seconds)
{
    if (!std::isfinite(seconds) || seconds <= 0.0) return;
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
        {
            // Where in the track the silence opens, in absolute seconds.  Only
            // the track holding the note is touched: a gap is an edit to one
            // part, not to the piece.
            auto at = -1.0;
            for (const auto& clip : track.clips)
                for (const auto& note : clip.notes)
                    if (note.id == noteId) at = clip.startSeconds + note.startSeconds;
            if (at < 0.0) continue;
            if (!changed) pushUndoLocked();
            changed = true;
            for (auto& clip : track.clips)
            {
                // A clip that begins after the point moves whole; the one the
                // note is in keeps its start and grows, with the notes from
                // there on carried along.
                if (clip.startSeconds >= at - 1.0e-9)
                {
                    clip.startSeconds += seconds;
                    continue;
                }
                const auto local = at - clip.startSeconds;
                auto moved = false;
                for (auto& note : clip.notes)
                    if (note.startSeconds >= local - 1.0e-9)
                    {
                        note.startSeconds += seconds;
                        moved = true;
                    }
                if (!moved) continue;
                auto needed = 0.0;
                for (const auto& note : clip.notes)
                    needed = std::max(needed, note.startSeconds + note.durationSeconds);
                clip.durationSeconds = std::max(needed, clip.durationSeconds + seconds);
            }
        }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::closeGapBeforeNote(const juce::String& noteId, double seconds)
{
    if (!std::isfinite(seconds) || seconds <= 0.0) return;
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
        {
            auto at = -1.0;
            for (const auto& clip : track.clips)
                for (const auto& note : clip.notes)
                    if (note.id == noteId) at = clip.startSeconds + note.startSeconds;
            if (at < 0.0) continue;
            // Never past the front of the piece: the silence being closed is
            // only as long as there is room to close it into.
            const auto span = std::min(seconds, at);
            if (span <= 0.0) continue;
            if (!changed) pushUndoLocked();
            changed = true;
            for (auto& clip : track.clips)
            {
                if (clip.startSeconds >= at - 1.0e-9)
                {
                    clip.startSeconds = std::max(0.0, clip.startSeconds - span);
                    continue;
                }
                const auto local = at - clip.startSeconds;
                auto moved = false;
                for (auto& note : clip.notes)
                    if (note.startSeconds >= local - 1.0e-9)
                    {
                        note.startSeconds = std::max(0.0, note.startSeconds - span);
                        moved = true;
                    }
                if (!moved) continue;
                auto needed = 0.0;
                for (const auto& note : clip.notes)
                    needed = std::max(needed, note.startSeconds + note.durationSeconds);
                clip.durationSeconds = std::max(0.01,
                    std::max(needed, clip.durationSeconds - span));
            }
        }
    }
    if (changed) sendChangeMessage();
}

std::vector<juce::String> ProjectModel::notesOverlapping(
    const juce::String& clipId, double fromSeconds, double toSeconds) const
{
    std::vector<juce::String> found;
    if (toSeconds <= fromSeconds) return found;
    const juce::ScopedLock guard(lock);
    for (const auto& track : project.tracks)
        for (const auto& clip : track.clips)
        {
            if (clip.id != clipId) continue;
            for (const auto& note : clip.notes)
            {
                // Touching end to end is not overlapping: a phrase pasted
                // against the one before it replaces nothing.
                const auto start = clip.startSeconds + note.startSeconds;
                if (start < toSeconds - 1.0e-9
                    && start + note.durationSeconds > fromSeconds + 1.0e-9)
                    found.push_back(note.id);
            }
        }
    return found;
}

std::vector<juce::String> ProjectModel::insertNotes(
    const juce::String& clipId, const std::vector<NoteData>& noteTemplates,
    double absoluteStartSeconds)
{
    std::vector<juce::String> inserted;
    if (noteTemplates.empty()) return inserted;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
                if (clip.id == clipId)
                {
                    if (clip.durationSeconds <= 0.0) return inserted;
                    const auto utauMode = track.pitchAlgorithm == PitchAlgorithm::utau;
                    // A UTAU phrase may extend the clip it lands in, and so
                    // may anything landing in a clip with no recording behind
                    // it -- that is a span of the timeline, not a length of
                    // audio, the same rule drawing a note follows.  A
                    // recording ends where its audio does, so there a note is
                    // still cropped to what can actually sound.
                    const auto mayStretch = utauMode || !clipIsRecording(clip);
                    const auto localOrigin = mayStretch
                        ? std::max(0.0, absoluteStartSeconds - clip.startSeconds)
                        : juce::jlimit(0.0, clip.durationSeconds,
                            absoluteStartSeconds - clip.startSeconds);
                    std::vector<NoteData> candidates;
                    candidates.reserve(noteTemplates.size());
                    for (const auto& noteTemplate : noteTemplates)
                    {
                        auto note = noteTemplate;
                        note.startSeconds = localOrigin
                            + std::max(0.0, noteTemplate.startSeconds);
                        if (mayStretch)
                        {
                            // Copy every musical and UTAU-specific field
                            // verbatim.  Only the new object's id changes.
                            candidates.push_back(std::move(note));
                            continue;
                        }
                        const auto remaining = clip.durationSeconds - note.startSeconds;
                        if (remaining < 0.01) continue;
                        note.durationSeconds = std::min(
                            std::max(0.01, noteTemplate.durationSeconds), remaining);
                        note.consonantSeconds = std::min(note.consonantSeconds,
                                                        note.durationSeconds);
                        for (auto& point : note.contour)
                            point.timeSeconds = juce::jlimit(0.0,
                                note.durationSeconds, point.timeSeconds);
                        for (auto& point : note.pitchControlPoints)
                            point.timeSeconds = juce::jlimit(0.0,
                                note.durationSeconds, point.timeSeconds);
                        for (auto& point : note.diffSingerPitchReference)
                            point.timeSeconds = juce::jlimit(0.0,
                                note.durationSeconds, point.timeSeconds);
                        for (auto& point : note.diffSingerPitchOffset)
                            point.timeSeconds = juce::jlimit(0.0,
                                note.durationSeconds, point.timeSeconds);
                        for (auto& marker : note.sibilantMarkers)
                            marker = juce::jlimit(0.0, note.durationSeconds, marker);
                        candidates.push_back(std::move(note));
                    }
                    if (candidates.empty()) return inserted;
                    pushUndoLocked();
                    if (utauMode)
                    {
                        const auto phraseStart = std::min_element(
                            candidates.begin(), candidates.end(),
                            [](const auto& left, const auto& right)
                            {
                                return left.startSeconds < right.startSeconds;
                            })->startSeconds;
                        auto phraseEnd = phraseStart;
                        for (const auto& note : candidates)
                            phraseEnd = std::max(phraseEnd,
                                note.startSeconds + note.durationSeconds);
                        const auto phraseDuration = std::max(0.01,
                                                            phraseEnd - phraseStart);
                        auto collisionStart = std::numeric_limits<double>::max();
                        for (const auto& existing : clip.notes)
                            if (existing.startSeconds < phraseEnd - 1.0e-9
                                && existing.startSeconds + existing.durationSeconds
                                    > phraseStart + 1.0e-9)
                                collisionStart = std::min(collisionStart,
                                                          existing.startSeconds);
                        if (collisionStart != std::numeric_limits<double>::max())
                        {
                            for (auto& existing : clip.notes)
                                if (existing.startSeconds >= collisionStart - 1.0e-9)
                                    existing.startSeconds += phraseDuration;
                            const auto align = collisionStart - phraseStart;
                            for (auto& note : candidates)
                                note.startSeconds += align;
                        }
                    }
                    else
                    {
                        candidates.front().connectedToPrevious = false;
                        candidates.back().connectedToNext = false;
                    }
                    inserted.reserve(candidates.size());
                    for (auto& note : candidates)
                    {
                        note.id = makeId("note");
                        inserted.push_back(note.id);
                        clip.notes.push_back(std::move(note));
                    }
                    std::stable_sort(clip.notes.begin(), clip.notes.end(),
                        [](const auto& left, const auto& right)
                        {
                            return left.startSeconds < right.startSeconds;
                        });
                    if (utauMode)
                        for (const auto& note : clip.notes)
                            clip.durationSeconds = std::max(clip.durationSeconds,
                                note.startSeconds + note.durationSeconds);
                    break;
                }
    }
    if (!inserted.empty()) sendChangeMessage();
    return inserted;
}

std::vector<juce::String> ProjectModel::insertNativeAudioClips(
    const juce::String& trackId, const std::vector<ClipData>& templates,
    const std::vector<NativeConnection>& connections, double atSeconds)
{
    std::vector<juce::String> inserted;
    if(templates.empty()||!std::isfinite(atSeconds)||atSeconds<0)return inserted;
    {
        const juce::ScopedLock guard(lock);
        const auto target=std::find_if(project.tracks.begin(),project.tracks.end(),[&](const auto& t){return t.id==trackId;});
        if(target==project.tracks.end()||!trackShowsAllNativeRegions(*target))return inserted;
        if(std::any_of(templates.begin(),templates.end(),[](const auto& c){return c.notes.empty()
            ||!c.sourceFile.existsAsFile()||!std::isfinite(c.durationSeconds)||c.durationSeconds<=0;}))return inserted;
        atSeconds=nativePasteStart(*target,templates,atSeconds);
        pushUndoLocked();
        std::map<juce::String,juce::String> noteIds;
        for(auto copy:templates)
        {
            copy.id=makeId("clip");copy.startSeconds+=atSeconds;
            for(auto& note:copy.notes)
            {
                const auto old=note.id;note.id=makeId("note");if(copy.parts.empty())note.clipPartId.clear();
                noteIds[old]=note.id;inserted.push_back(note.id);
            }
            target->clips.push_back(std::move(copy));
        }
        for(auto connection:connections)
        {
            const auto left=noteIds.find(connection.leftNoteId),right=noteIds.find(connection.rightNoteId);
            if(left==noteIds.end()||right==noteIds.end())continue;
            connection.id=makeId("connection");connection.leftNoteId=left->second;connection.rightNoteId=right->second;
            connection.boundarySeconds+=atSeconds;project.nativeConnections.push_back(std::move(connection));
        }
    }
    if(!inserted.empty())sendChangeMessage();
    return inserted;
}

std::vector<juce::String> ProjectModel::duplicateNotes(
    const std::vector<juce::String>& noteIds, const juce::String& targetClipId,
    double absoluteStartSeconds)
{
    const auto data = snapshot();
    const auto native = copyNativeAudioNotes(data, noteIds);
    if (!native.clips.empty())
    {
        juce::String targetTrack;
        for (const auto& track : data.tracks) for (const auto& clip : track.clips)
            if (clip.id == targetClipId || (targetClipId.isEmpty()
                && std::any_of(clip.notes.begin(),clip.notes.end(),[&](const auto& n){return n.id==noteIds.front();})))
                targetTrack=track.id;
        return insertNativeAudioClips(targetTrack,native.clips,native.connections,absoluteStartSeconds);
    }
    std::vector<std::pair<double, NoteData>> found;
    juce::String destination = targetClipId;
    for (const auto& track : data.tracks)
        for (const auto& clip : track.clips)
            for (const auto& note : clip.notes)
                if (std::find(noteIds.begin(), noteIds.end(), note.id) != noteIds.end())
                {
                    if (destination.isEmpty()) destination = clip.id;
                    found.emplace_back(clip.startSeconds + note.startSeconds, note);
                }
    if (found.empty() || destination.isEmpty()) return {};
    std::stable_sort(found.begin(), found.end(), [](const auto& left, const auto& right)
    {
        return left.first < right.first;
    });
    const auto origin = found.front().first;
    std::vector<NoteData> templates;
    templates.reserve(found.size());
    for (auto& [absolute, note] : found)
    {
        note.startSeconds = absolute - origin;
        templates.push_back(std::move(note));
    }
    return insertNotes(destination, templates, absoluteStartSeconds);
}

void ProjectModel::resizeNote(const juce::String& noteId, double newStart, double newDuration)
{
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
        {
            for (auto& clip : track.clips)
            {
                for (auto& note : clip.notes)
                {
                    if (note.id == noteId)
                    {
                        pushUndoLocked();
                        const auto oldStart = note.startSeconds;
                        const auto oldDuration = std::max(0.01, note.durationSeconds);
                        const auto oldEnd = oldStart + oldDuration;
                        auto otherEnd = 0.0;
                        for (const auto& candidate : clip.notes)
                            if (candidate.id != noteId)
                                otherEnd = std::max(otherEnd,
                                    candidate.startSeconds + candidate.durationSeconds);
                        const auto wasTail = oldEnd >= otherEnd - 1.0e-6
                            && oldEnd >= clip.durationSeconds - 0.002;
                        const auto targetDuration = std::max(0.01, newDuration);
                        const auto oldAttack = juce::jlimit(0.0, oldDuration,
                            note.consonantSeconds);
                        const auto newAttack = std::min(oldAttack, targetDuration);
                        const auto remapTime = [&](double time)
                        {
                            // A negative UTAU head anchor belongs to the
                            // preutterance before the nominal note.  Resizing
                            // the note body must not collapse it back to zero.
                            if (time < 0.0) return time;
                            const auto clamped = juce::jlimit(0.0, oldDuration, time);
                            if (clamped <= oldAttack || oldDuration <= oldAttack + 1.0e-9)
                                return oldAttack > 1.0e-9
                                    ? clamped * newAttack / oldAttack : 0.0;
                            return newAttack + (clamped - oldAttack)
                                * (targetDuration - newAttack) / (oldDuration - oldAttack);
                        };
                        for (auto& point : note.contour)
                            point.timeSeconds = remapTime(point.timeSeconds);
                        for (auto& point : note.pitchControlPoints)
                            point.timeSeconds = remapTime(point.timeSeconds);
                        for (auto& point : note.diffSingerPitchReference)
                            point.timeSeconds = remapTime(point.timeSeconds);
                        for (auto& point : note.diffSingerPitchOffset)
                            point.timeSeconds = remapTime(point.timeSeconds);
                        mapDiffSingerParameterTimes(note,remapTime);
                        for (auto& marker : note.sibilantMarkers)
                            marker = remapTime(marker);
                        note.startSeconds = std::max(0.0, newStart);
                        note.durationSeconds = targetDuration;
                        note.consonantSeconds = newAttack;
                        if (wasTail)
                            clip.durationSeconds = std::max(0.01,
                                std::max(otherEnd, note.startSeconds + note.durationSeconds));
                        changed = true;
                        break;
                    }
                }
                if (changed) break;
            }
            if (changed) break;
        }
    }
    if (changed) sendChangeMessage();
}

juce::String ProjectModel::splitNote(const juce::String& noteId, double localSeconds)
{
    juce::String createdId;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
        {
            for (auto& clip : track.clips)
            {
                for (std::size_t noteIndex = 0; noteIndex < clip.notes.size(); ++noteIndex)
                {
                    if (clip.notes[noteIndex].id != noteId) continue;
                    const auto original = clip.notes[noteIndex];
                    const auto split = juce::jlimit(0.01,
                        std::max(0.01, original.durationSeconds - 0.01), localSeconds);
                    if (split <= 0.0099 || split >= original.durationSeconds - 0.0099)
                        return {};

                    const auto evaluate = [&](double time)
                    {
                        PitchPoint result;
                        result.timeSeconds = time;
                        if (original.contour.empty()) return result;
                        const auto right = std::lower_bound(original.contour.begin(),
                            original.contour.end(), time,
                            [](const PitchPoint& point, double value)
                            {
                                return point.timeSeconds < value;
                            });
                        const auto rightIndex = static_cast<std::size_t>(right == original.contour.end()
                            ? original.contour.size() - 1 : right - original.contour.begin());
                        const auto leftIndex = rightIndex > 0
                            && original.contour[rightIndex].timeSeconds > time
                                ? rightIndex - 1 : rightIndex;
                        const auto& left = original.contour[leftIndex];
                        const auto& next = original.contour[rightIndex];
                        const auto amount = next.timeSeconds > left.timeSeconds
                            ? static_cast<float>(juce::jlimit(0.0, 1.0,
                                (time - left.timeSeconds)
                                    / (next.timeSeconds - left.timeSeconds))) : 0.0f;
                        result.relativeCents = left.relativeCents
                            + (next.relativeCents - left.relativeCents) * amount;
                        result.withoutVibratoCents = left.withoutVibratoCents
                            + (next.withoutVibratoCents - left.withoutVibratoCents) * amount;
                        result.voiced = left.voiced && next.voiced;
                        result.manualTargetCents = left.manualTargetCents
                            + (next.manualTargetCents - left.manualTargetCents) * amount;
                        result.hasManualTarget = left.hasManualTarget && next.hasManualTarget;
                        return result;
                    };

                    auto left = original;
                    auto right = original;
                    left.durationSeconds = split;
                    left.consonantSeconds = std::min(original.consonantSeconds, split);
                    left.connectedToNext = false;
                    right.id = makeId("note");
                    createdId = right.id;
                    right.startSeconds = original.startSeconds + split;
                    right.durationSeconds = original.durationSeconds - split;
                    right.consonantSeconds = original.consonantSeconds > split
                        ? original.consonantSeconds - split : 0.0;
                    right.connectedToPrevious = false;

                    if (trackShowsAllNativeRegions(track))
                    {
                        const auto sources = expandedClipParts(clip, true);
                        const auto source = std::find_if(sources.begin(), sources.end(), [&](const auto& part)
                            { return std::any_of(part.notes.begin(), part.notes.end(),
                                [&](const auto& value) { return value.id == noteId; }); });
                        if (source != sources.end())
                        {
                            bindNativeNoteSource(left, *source);
                            bindNativeNoteSource(right, *source);
                        }
                    }

                    left.contour.clear();
                    right.contour.clear();
                    for (const auto& point : original.contour)
                    {
                        if (point.timeSeconds < split - 1.0e-8)
                            left.contour.push_back(point);
                        if (point.timeSeconds > split + 1.0e-8)
                        {
                            auto shifted = point;
                            shifted.timeSeconds -= split;
                            right.contour.push_back(shifted);
                        }
                    }
                    auto boundary = evaluate(split);
                    boundary.timeSeconds = split;
                    left.contour.push_back(boundary);
                    boundary.timeSeconds = 0.0;
                    right.contour.insert(right.contour.begin(), boundary);

                    left.pitchControlPoints.clear();
                    right.pitchControlPoints.clear();
                    if (!original.pitchControlPoints.empty())
                    {
                        const auto controlPitchAt = [&](double time)
                        {
                            const auto found = std::upper_bound(
                                original.pitchControlPoints.begin(),
                                original.pitchControlPoints.end(), time,
                                [](double value, const PitchCurveEditPoint& point)
                                {
                                    return value < point.timeSeconds;
                                });
                            if (found == original.pitchControlPoints.begin())
                                return found->targetMidi;
                            if (found == original.pitchControlPoints.end())
                                return original.pitchControlPoints.back().targetMidi;
                            const auto& before = *(found - 1);
                            const auto amount = found->timeSeconds > before.timeSeconds
                                ? static_cast<float>((time - before.timeSeconds)
                                    / (found->timeSeconds - before.timeSeconds)) : 0.0f;
                            return before.targetMidi
                                + (found->targetMidi - before.targetMidi) * amount;
                        };
                        for (const auto& point : original.pitchControlPoints)
                        {
                            if (point.timeSeconds < split - 1.0e-8)
                                left.pitchControlPoints.push_back(point);
                            if (point.timeSeconds > split + 1.0e-8)
                            {
                                auto shifted = point;
                                shifted.timeSeconds -= split;
                                right.pitchControlPoints.push_back(shifted);
                            }
                        }
                        const auto boundaryPitch = controlPitchAt(split);
                        left.pitchControlPoints.push_back({ split, boundaryPitch });
                        right.pitchControlPoints.insert(right.pitchControlPoints.begin(),
                            { 0.0, boundaryPitch });
                    }

                    std::erase_if(left.utauFlagCurves,[](const auto& c){return isDiffSingerParameter(c.flag);});
                    std::erase_if(right.utauFlagCurves,[](const auto& c){return isDiffSingerParameter(c.flag);});
                    for(const auto& c:original.utauFlagCurves) if(isDiffSingerParameter(c.flag) && !c.flag.startsWith("DS:PTS:") && !c.points.empty()) {
                        FlagCurve a{c.flag,{}},b{c.flag,{}};
                        for(auto p:c.points) {
                            if(p.timeSeconds<split) a.points.push_back(p);
                            if(p.timeSeconds>split) {p.timeSeconds-=split;b.points.push_back(p);}
                        }
                        const auto value=flagCurveValueAt(c.points,split);
                        a.points.push_back({split,value});b.points.insert(b.points.begin(),{0,value});
                        left.utauFlagCurves.push_back(std::move(a));right.utauFlagCurves.push_back(std::move(b));
                    }
                    left.diffSingerPitchOffset.clear();right.diffSingerPitchOffset.clear();
                    if (!original.diffSingerPitchOffset.empty())
                    {
                        for (const auto& p : original.diffSingerPitchOffset)
                        {
                            if (p.timeSeconds<split) left.diffSingerPitchOffset.push_back(p);
                            if (p.timeSeconds>split) { auto q=p;q.timeSeconds-=split;right.diffSingerPitchOffset.push_back(q); }
                        }
                        const auto value=diffSingerPitchOffsetAt(original,split);
                        left.diffSingerPitchOffset.push_back({split,value,PitchCurveShape::linear});
                        right.diffSingerPitchOffset.insert(right.diffSingerPitchOffset.begin(),{0,value,PitchCurveShape::linear});
                    }
                    left.diffSingerPitchReference.clear();
                    right.diffSingerPitchReference.clear();
                    if (!original.diffSingerPitchReference.empty())
                    {
                        for (const auto& point : original.diffSingerPitchReference)
                        {
                            if (point.timeSeconds < split) left.diffSingerPitchReference.push_back(point);
                            if (point.timeSeconds > split)
                            {
                                auto shifted = point; shifted.timeSeconds -= split;
                                right.diffSingerPitchReference.push_back(shifted);
                            }
                        }
                        const auto midi = evaluatePitchCurve(original.diffSingerPitchReference, split);
                        left.diffSingerPitchReference.push_back({split, midi, PitchCurveShape::linear});
                        right.diffSingerPitchReference.insert(right.diffSingerPitchReference.begin(),
                            {0.0, midi, PitchCurveShape::linear});
                    }

                    left.nativeEnvelope=backend::slicedNativeEnvelope(original.nativeEnvelope,0,split/original.durationSeconds);
                    right.nativeEnvelope=backend::slicedNativeEnvelope(original.nativeEnvelope,split/original.durationSeconds,1);
                    left.amplitudeEnvelope.clear();
                    right.amplitudeEnvelope.clear();
                    if (!original.amplitudeEnvelope.empty())
                    {
                        const auto amplitudeAt = [&](double time)
                        {
                            const auto found = std::upper_bound(
                                original.amplitudeEnvelope.begin(),
                                original.amplitudeEnvelope.end(), time,
                                [](double value, const AmplitudeEnvelopePoint& point)
                                {
                                    return value < point.timeSeconds;
                                });
                            if (found == original.amplitudeEnvelope.begin())
                                return found->gainDb;
                            if (found == original.amplitudeEnvelope.end())
                                return original.amplitudeEnvelope.back().gainDb;
                            const auto& before = *(found - 1);
                            const auto span = found->timeSeconds - before.timeSeconds;
                            const auto amount = span > 1.0e-9
                                ? static_cast<float>(juce::jlimit(0.0, 1.0,
                                    (time - before.timeSeconds) / span)) : 0.0f;
                            return backend::envelopeDbBetween(before.gainDb,
                                found->gainDb, amount, before.linearToNext);
                        };
                        // Whether the stretch the cut lands in runs straight in
                        // amplitude: the half after the cut carries on that way.
                        const auto cutStretchIsLinear = [&]
                        {
                            const auto found = std::upper_bound(
                                original.amplitudeEnvelope.begin(),
                                original.amplitudeEnvelope.end(), split,
                                [](double value, const AmplitudeEnvelopePoint& point)
                                {
                                    return value < point.timeSeconds;
                                });
                            return found != original.amplitudeEnvelope.begin()
                                && found != original.amplitudeEnvelope.end()
                                && (found - 1)->linearToNext;
                        }();
                        for (const auto& point : original.amplitudeEnvelope)
                        {
                            if (point.timeSeconds < split - 1.0e-8)
                                left.amplitudeEnvelope.push_back(point);
                            if (point.timeSeconds > split + 1.0e-8)
                            {
                                auto shifted = point;
                                shifted.timeSeconds -= split;
                                right.amplitudeEnvelope.push_back(shifted);
                            }
                        }
                        const auto boundaryGain = amplitudeAt(split);
                        const auto explicitCut = std::any_of(original.amplitudeEnvelope.begin(), original.amplitudeEnvelope.end(),
                            [split](const auto& p) { return std::abs(p.timeSeconds - split) < 1.e-8; });
                        left.amplitudeEnvelope.push_back({ split, boundaryGain, cutStretchIsLinear, explicitCut });
                        right.amplitudeEnvelope.insert(right.amplitudeEnvelope.begin(),
                            { 0.0, boundaryGain, cutStretchIsLinear, explicitCut });
                    }

                    left.sibilantMarkers.clear();
                    right.sibilantMarkers.clear();
                    for (const auto marker : original.sibilantMarkers)
                        if (marker <= split) left.sibilantMarkers.push_back(marker);
                        else right.sibilantMarkers.push_back(marker - split);

                    pushUndoLocked();
                    if(trackShowsAllNativeRegions(track))rememberNativeTrimSources(clip);
                    clip.notes[noteIndex] = std::move(left);
                    clip.notes.insert(clip.notes.begin()
                        + static_cast<std::ptrdiff_t>(noteIndex + 1), std::move(right));
                    break;
                }
                if (createdId.isNotEmpty()) break;
            }
            if (createdId.isNotEmpty()) break;
        }
    }
    if (createdId.isNotEmpty()) sendChangeMessage();
    return createdId;
}

bool ProjectModel::canJoinNativeNotes(const juce::String& a,const juce::String& b) const
{
    const juce::ScopedLock guard(lock);
    for(const auto& track:project.tracks)for(const auto& clip:track.clips)
        if(planNativeNoteJoin(track,clip,a,b))return true;
    return false;
}

juce::String ProjectModel::joinNativeNotes(const juce::String& a,const juce::String& b)
{
    juce::String kept,removed;
    {
        const juce::ScopedLock guard(lock);
        for(auto& track:project.tracks)for(auto& clip:track.clips)
            if(kept.isEmpty())if(auto joined=planNativeNoteJoin(track,clip,a,b))
            {
                kept=std::any_of(joined->notes.begin(),joined->notes.end(),[&](const auto& n){return n.id==a;})?a:b;
                removed=kept==a?b:a;pushUndoLocked();clip=std::move(*joined);
            }
        if(kept.isEmpty())return {};
        std::erase_if(project.nativeConnections,[&](const auto& c){return(c.leftNoteId==kept&&c.rightNoteId==removed)||(c.leftNoteId==removed&&c.rightNoteId==kept);});
        for(auto& c:project.nativeConnections){if(c.leftNoteId==removed)c.leftNoteId=kept;if(c.rightNoteId==removed)c.rightNoteId=kept;}
    }
    sendChangeMessage();return kept;
}

juce::String ProjectModel::mergeNotes(const std::vector<juce::String>& noteIds)
{
    if (noteIds.size() < 2) return {};
    const auto includes = [&](const juce::String& id)
    {
        return std::find(noteIds.begin(), noteIds.end(), id) != noteIds.end();
    };

    juce::String mergedId;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
        {
            for (auto& clip : track.clips)
            {
                std::vector<std::size_t> selectedIndices;
                for (std::size_t index = 0; index < clip.notes.size(); ++index)
                    if (includes(clip.notes[index].id)) selectedIndices.push_back(index);
                // A merge is deliberately confined to one clip.  This also
                // verifies that every requested note still exists when the
                // asynchronous context-menu command is finally delivered.
                if (selectedIndices.size() != noteIds.size()) continue;

                std::stable_sort(selectedIndices.begin(), selectedIndices.end(),
                    [&](std::size_t left, std::size_t right)
                    {
                        return clip.notes[left].startSeconds
                            < clip.notes[right].startSeconds;
                    });
                const auto firstIndex = selectedIndices.front();
                const auto lastIndex = selectedIndices.back();
                auto merged = clip.notes[firstIndex];
                auto totalDuration = 0.0;
                for (const auto index : selectedIndices)
                    totalDuration += std::max(0.01, clip.notes[index].durationSeconds);

                merged.startSeconds = clip.notes[firstIndex].startSeconds;
                merged.durationSeconds = std::max(0.01, totalDuration);
                merged.label.clear();
                merged.connectedToPrevious = clip.notes[firstIndex].connectedToPrevious;
                merged.connectedToNext = clip.notes[lastIndex].connectedToNext;
                merged.consonantSeconds = std::min(merged.consonantSeconds,
                                                    merged.durationSeconds);
                // Curves from several independent notes use incompatible
                // local time origins.  Start the merged, lyric-less note as a
                // clean flat note instead of retaining misleading fragments.
                merged.contour.clear();
                merged.contour.push_back({ 0.0, 0.0f, 0.0f, true });
                merged.contour.push_back({ merged.durationSeconds, 0.0f, 0.0f, true });
                merged.pitchControlPoints.clear();
                merged.diffSingerPitchReference.clear();
                merged.diffSingerPitchOffset.clear();
                std::erase_if(merged.utauFlagCurves,[](const auto& c){return isDiffSingerParameter(c.flag);});
                merged.diffSingerPitchReferenceFromSavedPitch = false;
                merged.amplitudeEnvelope.clear();
                merged.nativeEnvelope={};
                merged.sibilantMarkers.clear();
                mergedId = merged.id;

                pushUndoLocked();
                clip.notes.erase(std::remove_if(clip.notes.begin(), clip.notes.end(),
                    [&](const auto& note) { return includes(note.id); }), clip.notes.end());
                clip.notes.push_back(std::move(merged));
                std::stable_sort(clip.notes.begin(), clip.notes.end(),
                    [](const auto& left, const auto& right)
                    {
                        return left.startSeconds < right.startSeconds;
                    });
                const auto mergedNote = std::find_if(clip.notes.begin(), clip.notes.end(),
                    [&](const auto& note) { return note.id == mergedId; });
                if (mergedNote != clip.notes.end())
                    clip.durationSeconds = std::max(clip.durationSeconds,
                        mergedNote->startSeconds + mergedNote->durationSeconds);
                break;
            }
            if (mergedId.isNotEmpty()) break;
        }
    }
    if (mergedId.isNotEmpty()) sendChangeMessage();
    return mergedId;
}

void ProjectModel::setNoteModulation(const juce::String& noteId, float modulation)
{
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
                for (auto& note : clip.notes)
                    if (note.id == noteId)
                    {
                        pushUndoLocked();
                        note.modulation = juce::jlimit(0.0f, 2.0f, modulation);
                        changed = true;
                        break;
                    }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::setNoteDrift(const juce::String& noteId, float drift)
{
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
                for (auto& note : clip.notes)
                    if (note.id == noteId)
                    {
                        pushUndoLocked();
                        note.drift = juce::jlimit(0.0f, 2.0f, drift);
                        changed = true;
                        break;
                    }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::setNoteTension(const juce::String& noteId, float tension)
{
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
                for (auto& note : clip.notes)
                    if (note.id == noteId)
                    {
                        pushUndoLocked();
                        note.tension = juce::jlimit(-1.0f, 1.0f, tension);
                        changed = true;
                        break;
                    }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::setNoteBreath(const juce::String& noteId, float breath)
{
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
                for (auto& note : clip.notes)
                    if (note.id == noteId)
                    {
                        pushUndoLocked();
                        note.breath = juce::jlimit(0.0f, 1.0f, breath);
                        changed = true;
                        break;
                    }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::setNoteFormant(const juce::String& noteId, float semitones)
{
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
                for (auto& note : clip.notes)
                    if (note.id == noteId)
                    {
                        pushUndoLocked();
                        note.formantSemitones = juce::jlimit(-12.0f, 12.0f, semitones);
                        changed = true;
                        break;
                    }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::setNoteGain(const juce::String& noteId, float gain)
{
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
                for (auto& note : clip.notes)
                    if (note.id == noteId)
                    {
                        pushUndoLocked();
                        note.gain = juce::jlimit(0.0f, 4.0f, gain);
                        changed = true;
                        break;
                    }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::setNoteAttack(const juce::String& noteId, double consonantSeconds,
                                 float attackSpeed)
{
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
                for (auto& note : clip.notes)
                    if (note.id == noteId)
                    {
                        pushUndoLocked();
                        note.consonantSeconds = juce::jlimit(0.0, note.durationSeconds,
                                                            consonantSeconds);
                        note.attackSpeed = juce::jlimit(0.05f, 20.0f, attackSpeed);
                        changed = true;
                        break;
                    }
    }
    if (changed) sendChangeMessage();
}

bool ProjectModel::setNotesTailFade(const std::vector<juce::String>& ids,int mode)
{
    if(mode<0||mode>2)return false;bool changed=false;
    {const juce::ScopedLock guard(lock);for(auto& track:project.tracks)
        if(track.pitchAlgorithm==PitchAlgorithm::utau&&!trackIsDiffSinger(track))
        for(auto& clip:track.clips)for(auto& note:clip.notes)
            if(std::find(ids.begin(),ids.end(),note.id)!=ids.end()&&note.utauTailFadeMode!=mode)
            {if(!changed)pushUndoLocked();note.utauTailFadeMode=mode;changed=true;}}
    if(changed)sendChangeMessage();return changed;
}

bool ProjectModel::setNotesTailFade(const std::vector<juce::String>& ids,int mode,const backend::TailFadeSettings& settings)
{
    if(mode<0||mode>2||!settings.valid())return false;bool changed=false;
    {const juce::ScopedLock guard(lock);for(auto& track:project.tracks)
        if(track.pitchAlgorithm==PitchAlgorithm::utau&&!trackIsDiffSinger(track))
        for(auto& clip:track.clips)for(auto& note:clip.notes)
            if(std::find(ids.begin(),ids.end(),note.id)!=ids.end()
                &&(note.utauTailFadeMode!=mode||note.utauTailFade!=settings))
            {if(!changed)pushUndoLocked();note.utauTailFadeMode=mode;note.utauTailFade=settings;changed=true;}}
    if(changed)sendChangeMessage();return changed;
}

bool ProjectModel::setNotesNativeEnvelope(const std::vector<juce::String>& ids,const backend::NativeEnvelopeSettings& settings)
{
    if(!settings.valid())return false;bool changed=false;
    {const juce::ScopedLock guard(lock);for(auto& track:project.tracks)if(trackShowsAllNativeRegions(track))
        for(auto& clip:track.clips)for(auto& note:clip.notes)
            if(std::find(ids.begin(),ids.end(),note.id)!=ids.end()&&note.nativeEnvelope!=settings)
            {if(!changed)pushUndoLocked();note.nativeEnvelope=settings;changed=true;}}
    if(changed)sendChangeMessage();return changed;
}

void ProjectModel::setNotesAmplitudeEnvelopeBase(
    const std::vector<juce::String>& noteIds, float basePercent)
{
    if (noteIds.empty()) return;
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        const auto next = juce::jlimit(0.0f, 200.0f, basePercent);
        auto targets = noteIds;
        for (const auto& track : project.tracks)
        {
            const auto groups = nativeSharedEnvelopes(track);
            for (const auto& id : noteIds)
                if (const auto p = groups.find(id); p != groups.end())
                    for (const auto& [otherId, other] : groups)
                        if (other.points == p->second.points && std::find(targets.begin(), targets.end(), otherId) == targets.end())
                            targets.push_back(otherId);
        }
        auto pushed = false;
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
                for (auto& note : clip.notes)
                {
                    if (std::find(targets.begin(), targets.end(), note.id) == targets.end())
                        continue;
                    if (std::abs(note.amplitudeEnvelopeBasePercent - next) <= 1.0e-6f) continue;
                    if (!pushed) { pushUndoLocked(); pushed = true; }
                    note.amplitudeEnvelopeBasePercent = next;
                    changed = true;
                }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::setNoteAttackSpeed(const juce::String& noteId, float attackSpeed)
{
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
                for (auto& note : clip.notes)
                    if (note.id == noteId)
                    {
                        const auto next = juce::jlimit(0.05f, 20.0f, attackSpeed);
                        if (std::abs(note.attackSpeed - next) <= 1.0e-6f) return;
                        pushUndoLocked();
                        note.attackSpeed = next;
                        changed = true;
                        break;
                    }
    }
    if (changed) sendChangeMessage();
}

namespace
{
// Changing a lyric is an HJPX edit. Material annotations are written only by
// the explicit material editor, never as a hidden side effect of note editing.
// This keeps UTAU OTO libraries read-only during ordinary project editing.
void relabelNote(NoteData& note, const juce::String& trimmed)
{
    if(note.materialAudioId.isNotEmpty()&&note.materialRegionId.isNotEmpty())
    {
        if(note.materialBoundAlias.isEmpty())note.materialBoundAlias=note.label;
        note.label=trimmed.isEmpty()?juce::String("-"):trimmed;
        note.materialLabelFollowsAlias=false;
        return; // explicit source identity survives a free display-label edit
    }
    const auto native = !note.nativeSegments.empty();
    const auto label = trimmed.isEmpty() ? juce::String("-") : trimmed;
    const auto oldLabel = note.label;
    note.label = label;
    note.diffSingerTiming.clear();
    note.diffSingerPronunciation.clear();
    // A local OTO/STP belongs to the previous alias even when the note also
    // carries native HJM segments. Keep the annotation but release the override.
    note.utauOto = {};
    note.utauStpSeconds = 0.0;
    if (native)
    {
        note.nativeRole = label == "_" ? NativeSegmentRole::transition
            : label == "-" ? NativeSegmentRole::unknown : NativeSegmentRole::vowel;
        for (auto& segment : note.nativeSegments)
            if (segment.role != NativeSegmentRole::transition
                && (segment.alias == oldLabel || segment.alias == "-"))
                segment.alias = label;
        return;
    }
    note.utauPreutteranceOverrideEnabled = false;
    note.utauPreutteranceSeconds = 0.0;
    note.utauOverlapOverrideEnabled = false;
    note.utauOverlapSeconds = 0.0;
    // An STP is tied to one recording. Changing the alias returns to the
    // selected voicebank entry instead of carrying the old recording offset.
    note.utauStpSeconds = 0.0;
    // A note's own oto is where things are in one recording too, and goes with
    // the rest.
    note.utauOto = {};
    note.utauJieSplitSet = false;
    note.utauJieSplit1 = 0.0;
    note.utauJieSplit2 = 0.0;
    note.utauJieSplit3 = 0.0;
}

// A note a 拼字 note leads into is sung from its own beat, whatever it is
// called.  Retyping its lyric is the first thing anyone does after putting one
// in front of it -- it becomes the vowel -- and relabelling releases the timing
// a note was pinned to, since another recording has its sound somewhere else.
// That released this pin as well, and the consonant was cut off before the
// beat again: heard to 0.96 instead of 1.04 on the check's entry.
//
// The preutterance goes back to nothing.  The overlap stays released, because
// that one does belong to the recording.
void keepVowelOnItsBeat(const ClipData& clip, NoteData& note)
{
    const auto ledIntoByPrefix = std::any_of(clip.notes.begin(), clip.notes.end(),
        [&note](const NoteData& other)
        {
            return other.id != note.id && other.durationSeconds <= 1.0e-12
                && std::abs(other.startSeconds - note.startSeconds) < 1.0e-9;
        });
    if (!ledIntoByPrefix) return;
    note.utauPreutteranceOverrideEnabled = true;
    note.utauPreutteranceSeconds = 0.0;
}
}

void ProjectModel::setNoteLabels(
    const std::vector<std::pair<juce::String, juce::String>>& labels)
{
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (const auto& [noteId, label] : labels)
        {
            const auto trimmed = label.trim().isEmpty() ? juce::String("-") : label.trim();
            for (auto& track : project.tracks)
                for (auto& clip : track.clips)
                    for (auto& note : clip.notes)
                        if (note.id == noteId && note.label != trimmed)
                        {
                            // Once for the whole batch, so it undoes as one.
                            if (!changed) pushUndoLocked();
                            relabelNote(note, trimmed);
                            if(note.materialAudioId.isEmpty()||note.materialRegionId.isEmpty())keepVowelOnItsBeat(clip, note);
                            changed = true;
                        }
        }
    }
    if (changed) sendChangeMessage();
}

int ProjectModel::convertTrackLyricsToPinyin(const juce::String& trackId)
{
    std::vector<std::pair<juce::String, juce::String>> labels;
    {
        const juce::ScopedLock guard(lock);
        for (const auto& track : project.tracks)
        {
            // Pinyin is what a UTAU voicebank's aliases are spelt in; a note
            // of any other kind of track is sung from its pitch, not its lyric.
            if (track.id != trackId) continue;
            for (const auto& clip : track.clips)
                for (const auto& note : clip.notes)
                {
                    const auto converted = lyricInPinyin(note.label);
                    if (converted != note.label) labels.emplace_back(note.id, converted);
                }
        }
    }
    // The same path a typed lyric takes, so each note is re-read against its
    // voicebank as if it had been typed -- and the whole lot undoes at once.
    setNoteLabels(labels);
    return static_cast<int>(labels.size());
}

void ProjectModel::setNoteLabel(const juce::String& noteId, const juce::String& label)
{
    const auto trimmed = label.trim().isEmpty() ? juce::String("-") : label.trim();
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
                for (auto& note : clip.notes)
                    if (note.id == noteId && note.label != trimmed)
                    {
                        pushUndoLocked();
                        relabelNote(note, trimmed);
                        if(note.materialAudioId.isEmpty()||note.materialRegionId.isEmpty())keepVowelOnItsBeat(clip, note);
                        changed = true;
                        break;
                    }
    }
    if (changed) sendChangeMessage();
}


void ProjectModel::setNoteUtauFlags(const juce::String& noteId, const juce::String& flags)
{
    const auto trimmed = flags.trim();
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
                for (auto& note : clip.notes)
                    if (note.id == noteId && note.utauFlags != trimmed)
                    {
                        pushUndoLocked();
                        note.utauFlags = trimmed;
                        changed = true;
                        break;
                    }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::setNotesUtauFlags(const std::vector<juce::String>& noteIds,
                                     const juce::String& flags)
{
    if (noteIds.empty()) return;
    const auto trimmed = flags.trim();
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
                for (auto& note : clip.notes)
                    if (std::find(noteIds.begin(), noteIds.end(), note.id) != noteIds.end()
                        && note.utauFlags != trimmed)
                    {
                        if (!changed) pushUndoLocked();
                        note.utauFlags = trimmed;
                        changed = true;
                    }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::setNotesUtauConsonantVelocity(
    const std::vector<juce::String>& noteIds, int velocity)
{
    if (noteIds.empty()) return;
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
                for (auto& note : clip.notes)
                    if (std::find(noteIds.begin(), noteIds.end(), note.id) != noteIds.end()
                        && note.utauConsonantVelocity != velocity)
                    {
                        if (!changed) pushUndoLocked();
                        if (track.pitchAlgorithm != PitchAlgorithm::utau
                            && note.consonantSeconds > 1.0e-6
                            && note.consonantSeconds < note.durationSeconds - 1.0e-6)
                        {
                            const auto oldVelocity = note.utauConsonantVelocity
                                == inheritedUtauConsonantVelocity ? 100 : note.utauConsonantVelocity;
                            const auto newVelocity = velocity == inheritedUtauConsonantVelocity
                                ? 100 : velocity;
                            const auto oldBoundary = note.consonantSeconds;
                            const auto newBoundary = juce::jlimit(0.001,
                                std::max(0.001, note.durationSeconds - 0.001),
                                oldBoundary * std::exp2(juce::jlimit(-20.0, 20.0,
                                    (static_cast<double>(oldVelocity) - newVelocity) / 100.0)));
                            const auto remapLocal = [&](double time)
                            {
                                if (time <= 0.0 || time >= note.durationSeconds) return time;
                                return time <= oldBoundary ? time * newBoundary / oldBoundary
                                    : newBoundary + (time - oldBoundary)
                                        * (note.durationSeconds - newBoundary)
                                        / (note.durationSeconds - oldBoundary);
                            };
                            if (clip.sourceTimeMap.empty())
                                clip.sourceTimeMap = { { 0.0, 0.0 },
                                    { clip.durationSeconds, clip.sourceDurationSeconds } };
                            const auto insertAnchor = [&](double time)
                            {
                                auto& map = clip.sourceTimeMap;
                                const auto right = std::lower_bound(map.begin(), map.end(), time,
                                    [](const auto& point, double t) { return point.targetSeconds < t; });
                                if (right == map.begin() || right == map.end()
                                    || std::abs(right->targetSeconds - time) < 1.0e-9) return;
                                const auto& left = *std::prev(right);
                                const auto u = (time-left.targetSeconds)/(right->targetSeconds-left.targetSeconds);
                                const auto source = left.sourceSeconds + u*(right->sourceSeconds-left.sourceSeconds);
                                map.insert(right, { time, source });
                            };
                            insertAnchor(note.startSeconds);
                            insertAnchor(note.startSeconds + oldBoundary);
                            insertAnchor(note.startSeconds + note.durationSeconds);
                            for (auto& point : clip.sourceTimeMap)
                                point.targetSeconds = note.startSeconds
                                    + remapLocal(point.targetSeconds - note.startSeconds);
                            for (auto& point : clip.nativeTrimClock)
                                point.targetSeconds = note.startSeconds + remapLocal(point.targetSeconds - note.startSeconds);
                            for (auto& point : note.contour) point.timeSeconds = remapLocal(point.timeSeconds);
                            for (auto& point : note.pitchControlPoints) point.timeSeconds = remapLocal(point.timeSeconds);
                            for (auto& point : note.diffSingerPitchReference) point.timeSeconds = remapLocal(point.timeSeconds);
                            for (auto& point : note.diffSingerPitchOffset) point.timeSeconds = remapLocal(point.timeSeconds);
                            mapDiffSingerParameterTimes(note,remapLocal);
                            for (auto& point : note.amplitudeEnvelope) point.timeSeconds = remapLocal(point.timeSeconds);
                            for (auto& marker : note.sibilantMarkers) marker = remapLocal(marker);
                            note.attackSpeed *= static_cast<float>(oldBoundary / newBoundary);
                            note.consonantSeconds = newBoundary;
                            note.utauPreutteranceSeconds = newBoundary;
                        }
                        note.utauConsonantVelocity = velocity;
                        changed = true;
                    }
    }
    if (changed) sendChangeMessage();
}

VibratoSpan vibratoSpanOf(const NoteData& note)
{
    const auto duration = std::max(0.0, note.vibratoReferenceDurationSeconds > 0
        ? note.vibratoReferenceDurationSeconds : note.durationSeconds);
    VibratoSpan span;
    span.end = duration * juce::jlimit(0.0, 100.0, note.vibratoEndPercent) / 100.0;
    span.start = std::max(0.0, span.end
        - duration * juce::jlimit(0.0, 100.0, note.vibratoLengthPercent) / 100.0);
    span.start = juce::jlimit(0.0, note.durationSeconds, span.start - note.vibratoTimeOffsetSeconds);
    span.end = juce::jlimit(0.0, note.durationSeconds, span.end - note.vibratoTimeOffsetSeconds);
    return span;
}

double vibratoCentsAt(const NoteData& note, double localSeconds)
{
    if (!note.vibratoEnabled) return 0.0;
    const auto duration = note.vibratoReferenceDurationSeconds > 0
        ? note.vibratoReferenceDurationSeconds : note.durationSeconds;
    if (duration <= 1.0e-9 || note.vibratoDepthCents == 0.0) return 0.0;
    localSeconds += note.vibratoTimeOffsetSeconds;
    // UTAU measures the vibrato span back from the end of the note; here from
    // where the vibrato ends, which is the note's end unless it was moved.
    VibratoSpan window;
    window.end = duration * juce::jlimit(0.0,100.0,note.vibratoEndPercent) / 100.0;
    window.start = std::max(0.0,window.end-duration*juce::jlimit(0.0,100.0,note.vibratoLengthPercent)/100.0);
    const auto start = window.start;
    const auto span = window.end - window.start;
    if (span <= 1.0e-9) return 0.0;
    if (localSeconds <= start) return 0.0;
    // Past its end the pitch is the note's own again.  At 100 the end is the
    // note's, and nothing reads past that.
    if (localSeconds > window.end + 1.0e-9) return 0.0;
    const auto position = juce::jlimit(0.0, 1.0, (localSeconds - start) / span);

    const auto fadeIn = juce::jlimit(0.0, 100.0, note.vibratoFadeInPercent) / 100.0;
    const auto fadeOut = juce::jlimit(0.0, 100.0, note.vibratoFadeOutPercent) / 100.0;
    auto envelope = 1.0;
    if (fadeIn > 1.0e-9) envelope = std::min(envelope, position / fadeIn);
    if (fadeOut > 1.0e-9) envelope = std::min(envelope, (1.0 - position) / fadeOut);
    envelope = juce::jlimit(0.0, 1.0, envelope);

    const auto cycleSeconds = std::max(0.01, note.vibratoCycleMs) / 1000.0;
    const auto phase = (localSeconds - start) / cycleSeconds
        + note.vibratoPhasePercent / 100.0;
    const auto swing = std::sin(phase * 2.0 * juce::MathConstants<double>::pi);
    const auto centre = note.vibratoOffsetPercent / 100.0;
    return note.vibratoDepthCents * envelope * (swing + centre);
}

void ProjectModel::setNotesUtauSplice(const std::vector<juce::String>& noteIds,
                                      bool enabled)
{
    if (noteIds.empty()) return;
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
                for (auto& note : clip.notes)
                    if (std::find(noteIds.begin(), noteIds.end(), note.id) != noteIds.end()
                        && note.utauSplice != enabled)
                    {
                        if (!changed) pushUndoLocked();
                        note.utauSplice = enabled;
                        changed = true;
                    }
    }
    if (changed) sendChangeMessage();
}

bool ProjectModel::resetCvvcBoundaries(const std::vector<juce::String>& noteIds,
                                       std::uint64_t expectedRevision)
{
    bool changed=false;
    {
        const juce::ScopedLock guard(lock);
        if(revision!=expectedRevision)return false;
        for(auto& track:project.tracks)
            if(track.chineseCvvc && track.utauMode==UtauMode::classic
                && trackUsesVoicebankSynthesis(track) && !trackIsDiffSinger(track))
                for(auto& clip:track.clips)for(auto& note:clip.notes)
                    if(std::find(noteIds.begin(),noteIds.end(),note.id)!=noteIds.end() && !note.cvvcBoundaries.empty())
                    {
                        if(!changed)pushUndoLocked();
                        note.cvvcBoundaries.clear();
                        note.cvvcPhonemeCache.clear();
                        changed=true;
                    }
    }
    if(changed)sendChangeMessage();
    return changed;
}

bool ProjectModel::setCvvcBoundary(const juce::String& noteId, int rightPart,
                                    double position, std::uint64_t expectedRevision)
{
    if ((rightPart!=2 && rightPart!=3) || !std::isfinite(position)) return false;
    bool changed=false;
    {
        const juce::ScopedLock guard(lock);
        if (revision!=expectedRevision) return false;
        for (auto& track:project.tracks)
            if (track.chineseCvvc && track.utauMode==UtauMode::classic
                && trackUsesVoicebankSynthesis(track) && !trackIsDiffSinger(track))
                for (auto& clip:track.clips) for (auto& note:clip.notes) if (note.id==noteId)
                {
                    const auto found=note.cvvcBoundaries.find(rightPart);
                    if (found!=note.cvvcBoundaries.end() && found->second==position) return false;
                    pushUndoLocked();
                    note.cvvcBoundaries[rightPart]=position;
                    note.cvvcPhonemeCache.clear();
                    changed=true;
                }
    }
    if (changed) sendChangeMessage();
    return changed;
}

bool ProjectModel::setCvvcEnvelope(const juce::String& noteId, int part, const juce::String& flag,
                                  backend::CvvcEnvelopeCurve points, std::uint64_t expectedRevision)
{
    return setCvvcEnvelopes({{noteId,part,flag,std::move(points)}},expectedRevision);
}

bool ProjectModel::setCvvcEnvelopes(std::vector<CvvcEnvelopeEdit> edits,std::uint64_t expectedRevision)
{
    for(auto& edit:edits)
    {
        if(edit.part<1||edit.part>3||edit.flag.startsWith("DS:"))return false;
        backend::CvvcEnvelopeEdits input;input[edit.part][edit.flag]=std::move(edit.points);
        auto normalized=backend::cvvcEnvelopesFromVar(backend::cvvcEnvelopesToVar(input));
        if(normalized.empty())return false;
        edit.points=std::move(normalized[edit.part][edit.flag]);
        if(edit.flag.isNotEmpty())for(auto& p:edit.points)
        {const auto& kind=flagCurveKindFor(edit.flag);p.value=juce::jlimit(kind.minimum,kind.maximum,p.value);}
    }
    bool changed=false;
    {
        const juce::ScopedLock guard(lock);
        if(revision!=expectedRevision)return false;
        for(auto& track:project.tracks)if(track.chineseCvvc&&track.utauMode==UtauMode::classic&&trackUsesVoicebankSynthesis(track))
            for(auto& clip:track.clips)for(auto& note:clip.notes)for(auto& edit:edits)if(note.id==edit.noteId)
            {
                const auto old=note.cvvcEnvelopes.find(edit.part);
                if(old!=note.cvvcEnvelopes.end())if(const auto found=old->second.find(edit.flag);
                    found!=old->second.end()&&found->second==edit.points&&(edit.flag.isEmpty()||note.utauFlagCurveEnabled))continue;
                if(!changed)pushUndoLocked();
                note.cvvcEnvelopes[edit.part][edit.flag]=std::move(edit.points);
                if(edit.flag.isNotEmpty())note.utauFlagCurveEnabled=true;
                changed=true;
            }
    }
    if(changed)sendChangeMessage();return changed;
}

void ProjectModel::setNotesUtauFlagCurveEnabled(
    const std::vector<juce::String>& noteIds, bool enabled)
{
    if (noteIds.empty()) return;
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
        {
            // 线性flag belongs to 界 and 谋.  Switching one off is allowed
            // anywhere: a track moved to plain UTAU with curves still on has to
            // be able to put them away.
            if (enabled && !trackTakesFlagCurves(track)) continue;
            for (auto& clip : track.clips)
                for (auto& note : clip.notes)
                    if (std::find(noteIds.begin(), noteIds.end(), note.id) != noteIds.end()
                        && note.utauFlagCurveEnabled != enabled)
                    {
                        if (!changed) pushUndoLocked();
                        note.utauFlagCurveEnabled = enabled;
                        // Nothing is written here.  This used to seed a flat
                        // two-point curve on g so the lane had something to
                        // drag; the lane now draws that starting line itself,
                        // for whichever flag is on show and across the stretch
                        // that flag reaches.  Seeding as well left g -- and
                        // only g -- opening with two handles instead of one,
                        // at the note's own start and end rather than the
                        // stretch it sounds for, and made a note count as
                        // carrying a curve before anything had been drawn.
                        changed = true;
                    }
        }
    }
    if (changed) sendChangeMessage();
}

bool ProjectModel::resetNotesUtauFlagCurves(
    const std::vector<juce::String>& noteIds)
{
    if (noteIds.empty()) return false;
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
                for (auto& note : clip.notes)
                    if (note.utauFlagCurveEnabled
                        && (!note.utauFlagCurves.empty() || backend::hasCvvcFlagCurve(note.cvvcEnvelopes))
                        && std::find(noteIds.begin(), noteIds.end(), note.id)
                               != noteIds.end())
                    {
                        if (!changed) pushUndoLocked();
                        std::erase_if(note.utauFlagCurves, [](const auto& c) { return !isDiffSingerParameter(c.flag); });
                        for(auto& [part, curves]:note.cvvcEnvelopes)std::erase_if(curves,[](const auto& c){return c.first.isNotEmpty();});
                        changed = true;
                    }
    }
    if (changed) sendChangeMessage();
    return changed;
}

bool ProjectModel::resetNotesUtauFlagCurve(
    const std::vector<juce::String>& noteIds, const juce::String& flag)
{
    if (noteIds.empty() || flag.isEmpty() || flag.startsWith("DS:REF:") || flag.startsWith("DS:PTS:")) return false;
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
                for (auto& note : clip.notes)
                {
                    if ((!note.utauFlagCurveEnabled && !flag.startsWith("DS:ABS:"))
                        || std::find(noteIds.begin(),noteIds.end(),note.id)==noteIds.end()) continue;
                    // HiFisampler may display a legacy unprefixed curve. Reset
                    // its native view without deleting the other engine's data.
                    if (flag.startsWith("HIFI:")) {
                        if (flagCurvePointsFor(note,flag).empty() && !backend::hasCvvcFlagCurve(note.cvvcEnvelopes,flag)) continue;
                    } else if (std::none_of(note.utauFlagCurves.begin(),note.utauFlagCurves.end(),
                        [&](const auto& curve) {return curve.flag==flag;}) && !backend::hasCvvcFlagCurve(note.cvvcEnvelopes,flag)) continue;
                    const auto keepLegacy = flag.startsWith("HIFI:")
                        && !flagCurvePointsFor(note,flag.substring(5)).empty();
                    if (!changed) pushUndoLocked();
                    std::erase_if(note.utauFlagCurves,[&](const auto& curve) {
                        return curve.flag==flag || (flag.startsWith("DS:ABS:") && curve.flag=="DS:PTS:"+flag.substring(7));
                    });
                    if (keepLegacy) note.utauFlagCurves.push_back({flag,{}});
                    for(auto& [part, curves]:note.cvvcEnvelopes)
                    {
                        curves.erase(flag);
                        if(flag.startsWith("HIFI:")&&curves.contains(flag.substring(5)))curves[flag]={};
                    }
                    changed = true;
                }
    }
    if (changed) sendChangeMessage();
    return changed;
}

bool ProjectModel::rememberDiffSingerParameters(const std::map<juce::String, std::vector<FlagCurve>>& curves)
{
    const juce::ScopedLock guard(lock);
    bool changed = false;
    for (auto& track : project.tracks) if (trackIsDiffSinger(track))
    for (auto& clip : track.clips) for (auto& note : clip.notes)
    {
        const auto found = curves.find(note.id);
        if (found == curves.end()) continue;
        for (const auto& incoming : found->second)
        {
            if (!incoming.flag.startsWith("DS:AUTO:") || incoming.points.empty()) continue;
            const auto code = incoming.flag.substring(8);
            if (std::none_of(diffSingerParameterKinds().begin(), diffSingerParameterKinds().end(),
                [&](const auto& k){return juce::String(k.flag)=="DS:ABS:"+code;})) continue;
            // Explicit predictions and edited bases own their original reference.
            if (std::any_of(note.utauFlagCurves.begin(), note.utauFlagCurves.end(), [&](const auto& c){
                return c.flag=="DS:ABS:"+code || c.flag=="DS:REF:"+code;
            })) continue;
            const auto& kind = flagCurveKindFor(incoming.flag);
            auto previous = -1.e30; bool valid = true;
            for (const auto& p : incoming.points) {
                valid = valid && std::isfinite(p.timeSeconds) && std::isfinite(p.value)
                    && p.timeSeconds>previous && p.timeSeconds>=-5 && p.timeSeconds<=60
                    && p.value>=kind.minimum && p.value<=kind.maximum;
                previous=p.timeSeconds;
            }
            if (!valid) continue;
            auto old=std::find_if(note.utauFlagCurves.begin(),note.utauFlagCurves.end(),
                [&](const auto& c){return c.flag==incoming.flag;});
            if (old!=note.utauFlagCurves.end() && old->points.size()==incoming.points.size()
                && std::equal(old->points.begin(),old->points.end(),incoming.points.begin(),
                    [](const auto& a,const auto& b){return a.timeSeconds==b.timeSeconds && a.value==b.value;})) continue;
            if (old==note.utauFlagCurves.end()) note.utauFlagCurves.push_back(incoming);
            else *old=incoming;
            changed=true;
        }
    }
    // Derived results make the document dirty, but must not consume an undo step.
    if (changed) { ++revision; sendChangeMessage(); }
    return changed;
}

bool ProjectModel::applyDiffSingerParameters(std::uint64_t expectedRevision,
    const std::map<juce::String,std::vector<FlagCurve>>& curves)
{
    const juce::ScopedLock guard(lock);
    if(revision!=expectedRevision || curves.empty()) return false;
    for(const auto& [id,items]:curves) for(const auto& c:items) {
        if(!c.flag.startsWith("DS:REF:") || c.points.empty()) return false;
        bool known=false;
        for(const auto& k:diffSingerParameterKinds()) known=known || c.flag=="DS:REF:"+juce::String(k.flag).substring(7);
        if(!known) return false;
        const auto& kind=flagCurveKindFor(c.flag);
        double previous=-1e30;
        for(const auto& p:c.points) {
            if(!std::isfinite(p.timeSeconds)||!std::isfinite(p.value)||p.timeSeconds<=previous
                || p.timeSeconds < -5 || p.timeSeconds>60 || p.value<kind.minimum || p.value>kind.maximum) return false;
            previous=p.timeSeconds;
        }
    }
    bool changed=false;
    for(auto& t:project.tracks) if(trackIsDiffSinger(t)) for(auto& c:t.clips) for(auto& n:c.notes) {
        auto found=curves.find(n.id); if(found==curves.end()) continue;
        if(!changed) pushUndoLocked(); changed=true;
        std::erase_if(n.utauFlagCurves,[](const auto& x){return isDiffSingerParameter(x.flag);});
        n.utauFlagCurves.insert(n.utauFlagCurves.end(),found->second.begin(),found->second.end());
    }
    if(changed) sendChangeMessage();
    return changed;
}

bool ProjectModel::setDiffSingerParameterCurve(const juce::String& id,const juce::String& key,
    std::vector<FlagCurvePoint> points)
{
    const juce::ScopedLock guard(lock);
    for(auto& t:project.tracks) if(trackIsDiffSinger(t)) for(auto& c:t.clips) for(auto& n:c.notes) {
        if(n.id!=id) continue;
        const auto& kind=flagCurveKindFor(key);
        for(auto& p:points) {
            if(!std::isfinite(p.timeSeconds)||!std::isfinite(p.value)) return false;
            if(!std::isfinite(p.bezierX1)||!std::isfinite(p.bezierY1)||!std::isfinite(p.bezierX2)||!std::isfinite(p.bezierY2)) return false;
            p.bezierX1=juce::jlimit(0.0f,1.0f,p.bezierX1);p.bezierX2=juce::jlimit(0.0f,1.0f,p.bezierX2);
            p.bezierY1=juce::jlimit(-2.0f,3.0f,p.bezierY1);p.bezierY2=juce::jlimit(-2.0f,3.0f,p.bezierY2);
            p.timeSeconds=juce::jlimit(-5.0,60.0,p.timeSeconds);
            p.value=juce::jlimit(kind.minimum,kind.maximum,p.value);
        }
        std::stable_sort(points.begin(),points.end(),[](auto& a,auto& b){return a.timeSeconds<b.timeSeconds;});
        points.erase(std::unique(points.begin(),points.end(),[](auto& a,auto& b){return std::abs(a.timeSeconds-b.timeSeconds)<1e-5;}),points.end());
        const auto previous=diffSingerParameterHandles(n,key);
        if(points.size()==previous.size() && std::equal(points.begin(),points.end(),previous.begin(),[](auto& a,auto& b){
            return a.timeSeconds==b.timeSeconds && a.value==b.value && a.shape==b.shape
                && a.bezierX1==b.bezierX1 && a.bezierY1==b.bezierY1 && a.bezierX2==b.bezierX2 && a.bezierY2==b.bezierY2;
        })) return false;
        auto full=spliceDiffSingerParameterHandles(n,key,points);
        pushUndoLocked();
        std::erase_if(n.utauFlagCurves,[&](const auto& x){return x.flag==key||x.flag=="DS:PTS:"+key.substring(7);});
        if(!points.empty()) {
            n.utauFlagCurves.push_back({key,std::move(full)});
            n.utauFlagCurves.push_back({"DS:PTS:"+key.substring(7),std::move(points)});
        }
        sendChangeMessage(); return true;
    }
    return false;
}

bool ProjectModel::setNoteUtauFlagCurve(const juce::String& noteId,
                                       const juce::String& flag,
                                       std::vector<FlagCurvePoint> points)
{
    if (noteId.isEmpty()) return false;
    if (flag.startsWith("DS:ABS:")) return setDiffSingerParameterCurve(noteId,flag,std::move(points));
    return setNoteUtauFlagCurves(flag, {{noteId, std::move(points)}}, std::nullopt, false);
}

bool ProjectModel::setNoteUtauFlagCurves(const juce::String& flag,
    const std::map<juce::String, std::vector<FlagCurvePoint>>& edits,
    std::optional<std::uint64_t> expectedRevision, bool enabledOnly)
{
    if (flag.isEmpty() || edits.empty() || flag.startsWith("DS:REF:") || flag.startsWith("DS:PTS:")) return false;
    std::map<juce::String, std::vector<FlagCurvePoint>> prepared;
    for (const auto& [id, input] : edits)
    {
        auto points = input;
        const auto& kind = flagCurveKindFor(flag);
        std::erase_if(points, [](const auto& point) { return !std::isfinite(point.timeSeconds) || !std::isfinite(point.value); });
        std::stable_sort(points.begin(), points.end(),
            [](const auto& left, const auto& right)
            {
                return left.timeSeconds < right.timeSeconds;
            });
        std::vector<FlagCurvePoint> normalized;
        normalized.reserve(points.size());
        for (auto point : points)
        {
            if (!std::isfinite(point.timeSeconds) || !std::isfinite(point.value)) continue;
            // A contour-wide shape means nothing to a single segment; it reads as a
            // straight line, so store it as one rather than leaving a value that
            // says something it cannot do.
            if (point.shape == PitchCurveShape::natural) point.shape = PitchCurveShape::linear;
            point.bezierX1 = juce::jlimit(0.0f, 1.0f, point.bezierX1);
            point.bezierX2 = juce::jlimit(0.0f, 1.0f, point.bezierX2);
            point.bezierY1 = juce::jlimit(-2.0f, 3.0f, point.bezierY1);
            point.bezierY2 = juce::jlimit(-2.0f, 3.0f, point.bezierY2);
            point.timeSeconds = juce::jlimit(-5.0, 60.0, point.timeSeconds);
            // The engine clamps this flag to that range either way; holding the
            // same one here keeps what is drawn and what is heard the same thing.
            point.value = juce::jlimit(kind.minimum, kind.maximum, point.value);
            // Two handles at the same instant would be a step the engine reads as
            // one; the later one wins, as it does there.
            if (!normalized.empty()
                && point.timeSeconds <= normalized.back().timeSeconds + 1.0e-5)
                normalized.back() = point;
            else
                normalized.push_back(point);
        }
        prepared.emplace(id, std::move(normalized));
    }
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        // A project edit arriving during a drag must not be overwritten by its preview.
        if (expectedRevision && revision != *expectedRevision) return false;
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
                for (auto& note : clip.notes)
                {
                    const auto found = prepared.find(note.id);
                    if (found == prepared.end() || (enabledOnly && !note.utauFlagCurveEnabled && !flag.startsWith("DS:ABS:"))) continue;
                    const auto& normalized = found->second;
                    if (!normalized.empty() && (!trackTakesFlagCurves(track)
                        || trackIsDiffSinger(track) != flag.startsWith("DS:"))) continue;
                    const auto existing = flagCurvePointsFor(note, flag);
                    const auto same = existing.size() == normalized.size()
                        && std::equal(existing.begin(),
                                      existing.end(), normalized.begin(),
                            [](const auto& left, const auto& right)
                            {
                                return std::abs(left.timeSeconds - right.timeSeconds) < 1.0e-9
                                    && std::abs(left.value - right.value) < 1.0e-6f
                                    && left.shape == right.shape
                                    && std::abs(left.bezierX1 - right.bezierX1) < 1.0e-6f
                                    && std::abs(left.bezierY1 - right.bezierY1) < 1.0e-6f
                                    && std::abs(left.bezierX2 - right.bezierX2) < 1.0e-6f
                                    && std::abs(left.bezierY2 - right.bezierY2) < 1.0e-6f;
                            });
                    if (same) continue;
                    if (!changed) pushUndoLocked();
                    std::erase_if(note.utauFlagCurves,
                        [&flag](const auto& curve) { return curve.flag == flag; });
                    if (flag.startsWith("DS:ABS:")) std::erase_if(note.utauFlagCurves,
                        [&](const auto& c) { return c.flag=="DS:PTS:"+flag.substring(7); });
                    // An empty native entry blocks legacy fallback, so clearing
                    // the lane really restores text/region flags after reopening.
                    if (!normalized.empty() || (flag.startsWith("HIFI:")
                        && !flagCurvePointsFor(note,flag.substring(5)).empty()))
                        note.utauFlagCurves.push_back({flag, normalized});
                    changed = true;
                }
    }
    if (changed) sendChangeMessage();
    return changed;
}

void ProjectModel::setNotesRegionFlags(const std::vector<juce::String>& noteIds,
                                       bool split, const juce::String& first,
                                       const juce::String& second,
                                       const juce::String& third,
                                       const juce::String& fourth)
{
    if (noteIds.empty()) return;
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
                for (auto& note : clip.notes)
                {
                    if (std::find(noteIds.begin(), noteIds.end(), note.id) == noteIds.end())
                        continue;
                    if (!changed) pushUndoLocked();
                    note.utauFlagSplit = split;
                    note.utauRegionFlags1 = first.trim();
                    note.utauRegionFlags2 = second.trim();
                    note.utauRegionFlags3 = third.trim();
                    note.utauRegionFlags4 = fourth.trim();
                    changed = true;
                }
    }
    if (changed) sendChangeMessage();
}

bool ProjectModel::bakeNoteVibratoIntoPitch(const juce::String& noteId)
{
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
                for (auto& note : clip.notes)
                {
                    if (note.id != noteId || !note.vibratoEnabled) continue;
                    if (note.durationSeconds <= 1.0e-9) continue;

                    // Base pitch under the swing: the edited curve if there is
                    // one, otherwise the note's own contour.
                    const auto baseAt = [&note](double time)
                    {
                        if (!note.pitchControlPoints.empty())
                            return evaluatePitchCurve(note.pitchControlPoints, time);
                        if (note.contour.empty()) return note.midiNote;
                        const auto cents = [&note](const PitchPoint& point)
                        {
                            return renderedPitchCents(note, point);
                        };
                        if (time <= note.contour.front().timeSeconds)
                            return note.midiNote + cents(note.contour.front()) / 100.0f;
                        if (time >= note.contour.back().timeSeconds)
                            return note.midiNote + cents(note.contour.back()) / 100.0f;
                        for (std::size_t i = 1; i < note.contour.size(); ++i)
                        {
                            const auto& left = note.contour[i - 1];
                            const auto& right = note.contour[i];
                            if (time > right.timeSeconds) continue;
                            const auto width = right.timeSeconds - left.timeSeconds;
                            const auto amount = width > 1.0e-9
                                ? static_cast<float>((time - left.timeSeconds) / width) : 0.0f;
                            return note.midiNote
                                + (cents(left) + (cents(right) - cents(left)) * amount) / 100.0f;
                        }
                        return note.midiNote + cents(note.contour.back()) / 100.0f;
                    };

                    // A vibrato is a sine, and from one crest to the next
                    // trough a sine is a single curve.  Sampling it eight times
                    // a cycle wrote a hundred points where a dozen would do,
                    // and every one of them then has to be dragged by hand.
                    //
                    // One point per crest and trough, joined by the cubic that
                    // fits a half sine, carries the same shape: handles at
                    // 0.36434 leave a worst-case error of 0.00019 of the swing,
                    // against 0.01 for the plain smooth join -- a fiftieth of a
                    // cent on a hundred-cent vibrato.
                    const auto cycle = std::max(0.02, note.vibratoCycleMs / 1000.0);
                    const auto swingSpan = vibratoSpanOf(note);
                    const auto swingStart = swingSpan.start;
                    const auto swingEnd = swingSpan.end;
                    const auto phase = note.vibratoPhasePercent / 100.0;

                    std::vector<double> times;
                    times.push_back(0.0);
                    if (swingStart > 1.0e-6) times.push_back(swingStart);
                    // A sine is at an extreme a quarter turn in, and every half
                    // turn after that; it crosses zero halfway between.
                    std::vector<double> extremes;
                    for (auto index = 0; index < 8000; ++index)
                    {
                        const auto time = swingStart
                            + cycle * (0.25 + index * 0.5 - phase);
                        if (time >= swingEnd) break;
                        if (time > swingStart + 1.0e-9)
                        {
                            times.push_back(time);
                            extremes.push_back(time);
                        }
                    }
                    // Between two extremes a half turn is one curve and needs
                    // nothing in between.  The stretch before the first and
                    // after the last is whatever the phase leaves over, so a
                    // crossing goes in it -- one point, two in all -- cutting
                    // it into quarter turns the fitted curves do cover.
                    const auto firstExtreme = extremes.empty()
                        ? swingEnd : extremes.front();
                    const auto lastExtreme = extremes.empty()
                        ? swingStart : extremes.back();
                    std::vector<double> crossings;
                    for (auto index = -4; index < 8000; ++index)
                    {
                        const auto time = swingStart + cycle * (index * 0.5 - phase);
                        if (time >= swingEnd - 1.0e-9) break;
                        if (time < swingStart - 1.0e-9) continue;
                        if (time < firstExtreme - 1.0e-9 || time > lastExtreme + 1.0e-9)
                        {
                            times.push_back(time);
                            crossings.push_back(time);
                        }
                    }
                    // The swing's own end, and -- when it stops before the
                    // note does -- the note's line straight after it and at
                    // the note's end, so what follows is the note's own pitch.
                    times.push_back(swingEnd);
                    if (swingEnd < note.durationSeconds - 1.0e-6)
                    {
                        times.push_back(std::min(note.durationSeconds, swingEnd + 0.001));
                        times.push_back(note.durationSeconds);
                    }
                    std::sort(times.begin(), times.end());
                    times.erase(std::unique(times.begin(), times.end(),
                        [](double left, double right)
                        {
                            return std::abs(left - right) < 1.0e-6;
                        }), times.end());

                    std::vector<PitchCurveEditPoint> points;
                    points.reserve(times.size());
                    for (const auto time : times)
                    {
                        PitchCurveEditPoint point;
                        point.timeSeconds = time;
                        point.targetMidi = baseAt(time)
                            + static_cast<float>(vibratoCentsAt(note, time) / 100.0);
                        // What the sine does between this point and the one
                        // before decides how they are joined.  Handles fitted
                        // to each arc: worst case 0.0002 of the swing, against
                        // 0.01 for a plain smooth join and 0.21 for a straight
                        // line -- a fiftieth of a cent on a hundred cent
                        // vibrato, against ten.
                        const auto kindOf = [&](double when)
                        {
                            const auto close = [when](double value)
                            {
                                return std::abs(value - when) < 1.0e-9;
                            };
                            if (std::any_of(extremes.begin(), extremes.end(), close))
                                return 1;   // crest or trough, where it is flat
                            if (std::any_of(crossings.begin(), crossings.end(), close))
                                return 2;   // zero crossing, where it is steepest
                            return 0;
                        };
                        const auto here = kindOf(time);
                        const auto before = points.empty()
                            ? 0 : kindOf(points.back().timeSeconds);
                        point.shape = PitchCurveShape::smooth;
                        if (before == 1 && here == 1)          // half a turn
                        {
                            point.shape = PitchCurveShape::customBezier;
                            point.bezierX1 = 0.36434f;
                            point.bezierY1 = 0.0f;
                            point.bezierX2 = 0.63566f;
                            point.bezierY2 = 1.0f;
                        }
                        else if (before == 2 && here == 1)     // rising quarter
                        {
                            point.shape = PitchCurveShape::customBezier;
                            point.bezierX1 = 0.33125f;
                            point.bezierY1 = 0.52033f;
                            point.bezierX2 = 0.64000f;
                            point.bezierY2 = 1.0f;
                        }
                        else if (before == 1 && here == 2)     // falling quarter
                        {
                            point.shape = PitchCurveShape::customBezier;
                            point.bezierX1 = 0.36000f;
                            point.bezierY1 = 0.0f;
                            point.bezierX2 = 0.66875f;
                            point.bezierY2 = 0.47967f;
                        }
                        points.push_back(point);
                    }

                    pushUndoLocked();
                    note.pitchControlPoints = std::move(points);
                    note.vibratoEnabled = false;
                    note.vibratoRealLine = false;
                    changed = true;
                }
    }
    if (changed) sendChangeMessage();
    return changed;
}

void ProjectModel::setNotesVibratoRealLine(const std::vector<juce::String>& noteIds,
                                           bool enabled)
{
    if (noteIds.empty()) return;
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
                for (auto& note : clip.notes)
                    if (std::find(noteIds.begin(), noteIds.end(), note.id) != noteIds.end()
                        && note.vibratoRealLine != enabled)
                    {
                        if (!changed) pushUndoLocked();
                        note.vibratoRealLine = enabled;
                        changed = true;
                    }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::setNotesVibrato(const std::vector<juce::String>& noteIds,
                                   const NoteData& parameters, bool enabled,
                                   std::optional<bool> realLine)
{
    if (noteIds.empty()) return;
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
                for (auto& note : clip.notes)
                {
                    if (note.nativeUnpitched || std::find(noteIds.begin(), noteIds.end(), note.id) == noteIds.end())
                        continue;
                    if (!changed) pushUndoLocked();
                    note.vibratoEnabled = enabled;
                    if (realLine) note.vibratoRealLine = *realLine;
                    note.vibratoLengthPercent = parameters.vibratoLengthPercent;
                    note.vibratoCycleMs = parameters.vibratoCycleMs;
                    note.vibratoDepthCents = parameters.vibratoDepthCents;
                    note.vibratoFadeInPercent = parameters.vibratoFadeInPercent;
                    note.vibratoFadeOutPercent = parameters.vibratoFadeOutPercent;
                    note.vibratoPhasePercent = parameters.vibratoPhasePercent;
                    note.vibratoOffsetPercent = parameters.vibratoOffsetPercent;
                    note.vibratoEndPercent = juce::jlimit(1.0, 100.0, parameters.vibratoEndPercent);
                    changed = true;
                }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::setNotesUtauJieSplitCleared(const std::vector<juce::String>& noteIds)
{
    if (noteIds.empty()) return;
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
                for (auto& note : clip.notes)
                    if (std::find(noteIds.begin(), noteIds.end(), note.id) != noteIds.end()
                        && note.utauJieSplitSet)
                    {
                        if (!changed) pushUndoLocked();
                        note.utauJieSplitSet = false;
                        note.utauJieSplit1 = 0.0;
                        note.utauJieSplit2 = 0.0;
                        note.utauJieSplit3 = 0.0;
                        changed = true;
                    }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::setNotesUtauJieSplit(const std::vector<juce::String>& noteIds,
                                        double first, double second, double third)
{
    if (noteIds.empty()) return;
    // Keep the three boundaries ordered inside the note; a drag may collapse a
    // region to nothing but must never invert one.
    const auto third_ = juce::jlimit(0.0, 1.0, third);
    const auto second_ = juce::jlimit(0.0, third_, second);
    const auto first_ = juce::jlimit(0.0, second_, first);
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
                for (auto& note : clip.notes)
                    if (std::find(noteIds.begin(), noteIds.end(), note.id) != noteIds.end()
                        && (!note.utauJieSplitSet
                            || note.utauJieSplit1 != first_
                            || note.utauJieSplit2 != second_
                            || note.utauJieSplit3 != third_))
                    {
                        if (!changed) pushUndoLocked();
                        note.utauJieSplitSet = true;
                        note.utauJieSplit1 = first_;
                        note.utauJieSplit2 = second_;
                        note.utauJieSplit3 = third_;
                        changed = true;
                    }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::clearNotesUtauJieSplit(const std::vector<juce::String>& noteIds)
{
    if (noteIds.empty()) return;
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
                for (auto& note : clip.notes)
                    if (std::find(noteIds.begin(), noteIds.end(), note.id) != noteIds.end()
                        && note.utauJieSplitSet)
                    {
                        if (!changed) pushUndoLocked();
                        note.utauJieSplitSet = false;
                        changed = true;
                    }
    }
    if (changed) sendChangeMessage();
}

bool ProjectModel::setNoteUtauOtoLocked(const juce::String& noteId,const backend::UtauOtoOverride& oto)
{
    bool changed=false;
    for(auto& track:project.tracks)for(auto& clip:track.clips)for(auto& note:clip.notes)
        if(note.id==noteId&&note.utauOto!=oto)
        {
            if(!changed)pushUndoLocked();
            note.utauOto=oto;changed=true;
        }
    return changed;
}

void ProjectModel::setNoteUtauOto(const juce::String& noteId,const backend::UtauOtoOverride& oto)
{
    bool changed=false;
    {const juce::ScopedLock guard(lock);changed=setNoteUtauOtoLocked(noteId,oto);}
    if(changed)sendChangeMessage();
}

bool ProjectModel::setNoteUtauOtoChecked(const juce::String& noteId,std::uint64_t expectedRevision,
    const backend::UtauOtoOverride& value,std::uint64_t& resultingRevision,juce::String& error)
{
    error.clear();resultingRevision=0;bool changed=false;
    {
        const juce::ScopedLock guard(lock);
        if(revision!=expectedRevision){error="The project changed while the local OTO editor was open.";return false;}
        if(noteId.isEmpty()){error="A stable note identity is required.";return false;}
        const TrackData* owner=nullptr;int matches=0;
        for(const auto& track:project.tracks)for(const auto& clip:track.clips)for(const auto& note:clip.notes)
            if(note.id==noteId){++matches;owner=&track;}
        if(matches!=1){error=matches?"Ambiguous note identity; no local OTO was changed.":"The note no longer exists.";return false;}
        if(owner->accompaniment||!owner->compose||effectiveTrackType(*owner)!=TrackType::utau||owner->utauMode!=UtauMode::classic)
        {error="This callback supports classic UTAU scalar OTO timing only, not native, DiffSinger or regional tracks.";return false;}
        if(value.enabled&&(value.hasRegions||value.classes.isNotEmpty()))
        {error="Classic OTO cannot apply regional source geometry or classes.";return false;}
        for(const auto number:{value.offsetMs,value.consonantMs,value.cutoffMs,value.preutteranceMs,value.overlapMs,value.onsetMs,value.glideMs,value.nucleusMs})
            if(!std::isfinite(number)){error="Local OTO values must be finite.";return false;}
        changed=setNoteUtauOtoLocked(noteId,value);
        resultingRevision=revision;
    }
    if(changed)sendChangeMessage();return true;
}

void ProjectModel::clearNotesUtauOto(const std::vector<juce::String>& noteIds)
{
    if (noteIds.empty()) return;
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
                for (auto& note : clip.notes)
                    if (note.utauOto.enabled
                        && std::find(noteIds.begin(), noteIds.end(), note.id) != noteIds.end())
                    {
                        if (!changed) pushUndoLocked();
                        note.utauOto = {};
                        changed = true;
                    }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::setNotesUtauStp(const std::vector<juce::String>& noteIds,
                                   double seconds)
{
    if (noteIds.empty() || !std::isfinite(seconds)) return;
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
                for (auto& note : clip.notes)
                {
                    if (std::find(noteIds.begin(), noteIds.end(), note.id)
                        == noteIds.end())
                        continue;
                    if (std::abs(note.utauStpSeconds - seconds) <= 1.0e-12) continue;
                    if (!changed) pushUndoLocked();
                    note.utauStpSeconds = seconds;
                    changed = true;
                }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::setNoteUtauTimingOverrides(
    const juce::String& noteId, bool enabled,
    double preutteranceSeconds, double overlapSeconds)
{
    if (!std::isfinite(preutteranceSeconds) || !std::isfinite(overlapSeconds))
        return;
    const auto normalizedPreutterance = std::max(0.0, preutteranceSeconds);
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
                for (auto& note : clip.notes)
                    if (note.id == noteId
                        && (note.utauPreutteranceOverrideEnabled != enabled
                            || note.utauOverlapOverrideEnabled != enabled
                            || (enabled && (std::abs(note.utauPreutteranceSeconds
                                                    - normalizedPreutterance) > 1.0e-9
                                || std::abs(note.utauOverlapSeconds
                                            - overlapSeconds) > 1.0e-9))))
                    {
                        pushUndoLocked();
                        note.utauPreutteranceOverrideEnabled = enabled;
                        note.utauPreutteranceSeconds = normalizedPreutterance;
                        note.utauOverlapOverrideEnabled = enabled;
                        note.utauOverlapSeconds = overlapSeconds;
                        changed = true;
                        break;
                    }
    }
    if (changed) sendChangeMessage();
}

bool ProjectModel::rememberCvvcPhonemes(const std::vector<std::pair<juce::String,juce::String>>& caches,
                                      std::uint64_t expectedRevision)
{
    const juce::ScopedLock guard(lock);
    if(revision!=expectedRevision)return false;
    bool changed=false;
    for(auto& track:project.tracks)if(track.chineseCvvc&&track.utauMode==UtauMode::classic)
        for(auto& clip:track.clips)for(auto& note:clip.notes)
            for(const auto& [id,text]:caches)if(note.id==id&&note.cvvcPhonemeCache!=text
                &&!backend::readCvvcPhonemeCache(text).spans.empty())
            {note.cvvcPhonemeCache=text;changed=true;}
    // Rendering is not a user edit: do not advance revision, push undo, or
    // trigger another render. The receiving view refreshes its own snapshot.
    return changed;
}

int ProjectModel::crossfadeNotes(std::vector<CrossfadeSpan> spans, std::uint64_t expectedRevision, bool curved)
{
    const juce::ScopedLock guard(lock);
    if (revision != expectedRevision || spans.size() < 2) return 0;
    struct Owner { NoteData* note; double start; const TrackData* track; };
    std::map<juce::String, Owner> owners;
    for (auto& track : project.tracks) for (auto& clip : track.clips) for (auto& note : clip.notes)
        for (const auto& span : spans) if (span.noteId == note.id)
            owners.emplace(note.id, Owner{&note, clip.startSeconds + note.startSeconds, &track});
    if (owners.size() != 2 || owners.begin()->second.track != owners.rbegin()->second.track) return 0;
    for (const auto& span : spans)
    {
        if (!owners.contains(span.noteId) || !std::isfinite(span.from) || !std::isfinite(span.to)
            || span.to - span.from < 1.e-6 || span.points.size() < 2 || span.part < 0 || span.part > 3) return 0;
        double last = -std::numeric_limits<double>::infinity();
        for (const auto& p : span.points)
        {
            if (!std::isfinite(p.timeSeconds) || !std::isfinite(p.gainDb) || p.timeSeconds < last) return 0;
            last = p.timeSeconds;
        }
    }
    std::stable_sort(spans.begin(), spans.end(), [](const auto& a, const auto& b) { return a.from < b.from; });
    const auto ramp = [curved](auto& points, double from, double to, bool rising)
    {
        const auto level = nativeEnvelopeDbAt(points, rising ? to : from);
        std::erase_if(points, [&](const auto& p) { return p.timeSeconds >= from - 1.e-9 && p.timeSeconds <= to + 1.e-9; });
        points.push_back({from, rising ? -60.f : level, true, true});
        if (curved)
        {
            // Persist in the existing amplitude-point format so every backend,
            // the lane and undo/save consume exactly the same shape. 32 linear
            // intervals approximate CVVC's complementary smoothstep to <0.001
            // in normalized gain, without changing legacy dB interpolation.
            backend::UtauMixRamp shape; shape.length=32;shape.rising=rising;shape.smoothCurve=true;
            const auto peak=backend::envelopeGainFromDb(level);
            for(int i=1;i<shape.length;++i)
                points.push_back({from+(to-from)*i/shape.length,
                    backend::envelopeDbFromGain(peak*shape.gainAt(i)),true,true});
        }
        points.push_back({to, rising ? level : -60.f, true, true});
        std::stable_sort(points.begin(), points.end(), [](const auto& a, const auto& b) { return a.timeSeconds < b.timeSeconds; });
    };
    std::set<std::size_t> edited;
    int count = 0;
    for (std::size_t a = 0; a < spans.size(); ++a) for (std::size_t b = a + 1; b < spans.size(); ++b)
    {
        auto& left = spans[a]; auto& right = spans[b];
        const auto end = left.soundingEnd > left.from ? std::min(left.to,left.soundingEnd) : left.to;
        const auto rightEnd = right.soundingEnd > right.from ? std::min(right.to,right.soundingEnd) : right.to;
        if (left.noteId == right.noteId || right.from >= end - 1.e-6) continue;
        // A handoff requires a preceding tail and a following head. A fully
        // nested sound is a layer, not a seam; fading it would mute the outer tail.
        if (right.from <= left.from + 1.e-6 || rightEnd <= end + 1.e-6) continue;
        ramp(left.points, right.from, end, false);
        ramp(right.points, right.from, end, true);
        edited.insert(a); edited.insert(b); ++count;
    }
    if (!count) return 0;
    bool changed = false;
    for (const auto index : edited)
    {
        const auto& span = spans[index]; auto& owner = owners.at(span.noteId);
        if (span.part)
        {
            backend::CvvcEnvelopeCurve curve;
            for (const auto& p : span.points) curve.push_back({(p.timeSeconds-span.from)/(span.to-span.from),p.gainDb,p.linearToNext});
            const auto part = owner.note->cvvcEnvelopes.find(span.part);
            if (part != owner.note->cvvcEnvelopes.end())
                if (const auto old = part->second.find(juce::String()); old != part->second.end() && old->second == curve) continue;
            if (!changed) pushUndoLocked(); changed = true;
            owner.note->cvvcEnvelopes[span.part][juce::String()] = std::move(curve);
        }
        else
        {
            auto points = span.points;
            for (auto& p : points) p.timeSeconds -= owner.start;
            const auto& old = owner.note->amplitudeEnvelope;
            if (old.size() == points.size() && std::equal(old.begin(),old.end(),points.begin(),[](const auto& a,const auto& b) {
                return std::abs(a.timeSeconds-b.timeSeconds)<1.e-9 && std::abs(a.gainDb-b.gainDb)<1.e-5f
                    && a.linearToNext==b.linearToNext && a.nativeSeamAnchor==b.nativeSeamAnchor; })) continue;
            if (!changed) pushUndoLocked(); changed = true;
            owner.note->amplitudeEnvelope = std::move(points);
        }
    }
    if (changed) sendChangeMessage();
    return count;
}

bool ProjectModel::setNoteAmplitudeEnvelope(
    const juce::String& noteId, std::vector<AmplitudeEnvelopePoint> points)
{
    std::vector<std::pair<juce::String, std::vector<AmplitudeEnvelopePoint>>> envelopes;
    envelopes.emplace_back(noteId, std::move(points));
    return setNotesAmplitudeEnvelopes(std::move(envelopes));
}

bool ProjectModel::setNotesAmplitudeEnvelopes(
    std::vector<std::pair<juce::String, std::vector<AmplitudeEnvelopePoint>>> envelopes)
{
    std::erase_if(envelopes, [](auto& entry)
    {
        auto& points = entry.second;
        std::stable_sort(points.begin(), points.end(), [](const auto& left, const auto& right)
        {
            return left.timeSeconds < right.timeSeconds;
        });
        std::vector<AmplitudeEnvelopePoint> normalized;
        normalized.reserve(points.size());
        for (auto point : points)
        {
            if (!std::isfinite(point.timeSeconds) || !std::isfinite(point.gainDb)) continue;
            point.timeSeconds = juce::jlimit(-5.0, 60.0, point.timeSeconds);
            point.gainDb = juce::jlimit(-60.0f, 12.0f, point.gainDb);
            if (!normalized.empty()
                && point.timeSeconds <= normalized.back().timeSeconds + 1.0e-5)
                continue;
            normalized.push_back(point);
        }
        points = std::move(normalized);
        return entry.first.isEmpty() || points.size() < 2;
    });
    if (envelopes.empty()) return false;

    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
                for (auto& note : clip.notes)
                    if (const auto entry = std::find_if(envelopes.begin(), envelopes.end(),
                            [&](const auto& value) { return value.first == note.id; });
                        entry != envelopes.end())
                    {
                        const auto& points = entry->second;
                        const auto hasPartEnvelope = std::any_of(note.cvvcEnvelopes.begin(),note.cvvcEnvelopes.end(),
                            [](const auto& part){return part.second.contains(juce::String());});
                        const auto same = !hasPartEnvelope && note.amplitudeEnvelope.size() == points.size()
                            && std::equal(note.amplitudeEnvelope.begin(),
                                          note.amplitudeEnvelope.end(), points.begin(),
                                [](const auto& left, const auto& right)
                                {
                                    return std::abs(left.timeSeconds - right.timeSeconds) <= 1.0e-7
                                        && std::abs(left.gainDb - right.gainDb) <= 1.0e-4f
                                        && left.linearToNext == right.linearToNext
                                        && left.nativeSeamAnchor == right.nativeSeamAnchor;
                                });
                        if (same) continue;
                        if (!changed) pushUndoLocked();
                        note.amplitudeEnvelope = points;
                        for(auto& [part, curves]:note.cvvcEnvelopes)curves.erase(juce::String());
                        changed = true;
                    }
    }
    if (changed) sendChangeMessage();
    return changed;
}


void ProjectModel::setNoteRobustPitchCurve(const juce::String& noteId, bool enabled)
{
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
                for (auto& note : clip.notes)
                    if (note.id == noteId && note.robustPitchCurve != enabled)
                    {
                        pushUndoLocked();
                        note.robustPitchCurve = enabled;
                        changed = true;
                        break;
                    }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::setDiffSingerOptions(const juce::String& trackId,
    const juce::String& language, const juce::String& speaker)
{
    const juce::ScopedLock guard(lock);
    for (auto& track : project.tracks)
        if (track.id == trackId && (track.diffSingerLanguage != language || track.diffSingerSpeaker != speaker))
        {
            pushUndoLocked();
            track.diffSingerLanguage = language;
            track.diffSingerSpeaker = speaker;
            sendChangeMessage();
            break;
        }
}

bool ProjectModel::applyDiffSingerPronunciation(std::uint64_t expectedRevision, const juce::String& trackId,
    const std::vector<std::pair<juce::String, juce::String>>& readings, const juce::String& dictionary)
{
    const juce::ScopedLock guard(lock);
    if (revision != expectedRevision) return false;
    auto track = std::find_if(project.tracks.begin(), project.tracks.end(),
        [&](const auto& item) { return item.id == trackId && trackIsDiffSinger(item); });
    if (track == project.tracks.end()) return false;
    std::vector<std::pair<NoteData*, juce::String>> changes;
    for (const auto& [id, value] : readings)
    {
        NoteData* found = nullptr;
        for (auto& clip : track->clips) for (auto& note : clip.notes) if (note.id == id) found = &note;
        if (found == nullptr) return false;
        if (found->diffSingerPronunciation != value.trim()) changes.emplace_back(found, value.trim());
    }
    if (changes.empty() && track->diffSingerDictionary == dictionary) return true;
    pushUndoLocked();
    if (track->diffSingerDictionary != dictionary)
        for (auto& clip : track->clips) for (auto& note : clip.notes) note.diffSingerTiming.clear();
    track->diffSingerDictionary = dictionary;
    for (const auto& [note, text] : changes)
    { note->diffSingerPronunciation = text; note->diffSingerTiming.clear(); }
    sendChangeMessage();
    return true;
}

bool ProjectModel::applyDiffSingerTiming(std::uint64_t expectedRevision,
    const std::vector<std::pair<juce::String, juce::String>>& timings)
{
    const juce::ScopedLock guard(lock);
    if (revision != expectedRevision || timings.empty()) return false;
    std::vector<std::pair<NoteData*, juce::String>> changes;
    for (const auto& [id, text] : timings)
    {
        if (text.isNotEmpty())
        {
            const auto value = juce::JSON::parse(text);
            if (!value.isObject() || value["context"].toString().isEmpty()
                || !value["tokens"].isArray() || !value["starts"].isArray()
                || value["tokens"].size() != value["starts"].size()) return false;
            for (const auto& at : *value["starts"].getArray())
                if (!at.isVoid() && (!(at.isDouble() || at.isInt() || at.isInt64())
                    || !std::isfinite((double) at))) return false;
        }
        NoteData* found = nullptr;
        for (auto& track : project.tracks) if (trackIsDiffSinger(track))
            for (auto& clip : track.clips) for (auto& note : clip.notes)
                if (note.id == id) found = &note;
        if (found == nullptr) return false;
        if (found->diffSingerTiming != text) changes.emplace_back(found, text);
    }
    if (changes.empty()) return true;
    pushUndoLocked();
    for (const auto& [note, text] : changes) note->diffSingerTiming = text;
    sendChangeMessage();
    return true;
}

bool ProjectModel::restoreDiffSingerPitchPoint(std::uint64_t expectedRevision,
    const juce::String& noteId, const std::vector<PitchCurveEditPoint>& displayedPoints, int index)
{
    const juce::ScopedLock guard(lock);
    if (revision != expectedRevision) return false;
    for (const auto& p : displayedPoints)
        if (!std::isfinite(p.timeSeconds) || !std::isfinite(p.targetMidi)) return false;
    for (auto& track : project.tracks) if (trackIsDiffSinger(track))
        for (auto& clip : track.clips) for (auto& note : clip.notes)
            if (note.id == noteId)
            {
                const auto restored = restoredDiffSingerPitchPoint(note, displayedPoints, index);
                return !restored.empty() && setNotePitchCurve(noteId, restored, true);
            }
    return false;
}

bool ProjectModel::applyDiffSingerPitch(std::uint64_t expectedRevision,
    const std::vector<std::pair<juce::String, std::vector<PitchCurveEditPoint>>>& curves,
    bool preserveOutsideNoteHandles)
{
    const juce::ScopedLock guard(lock);
    if (revision != expectedRevision || curves.empty()) return false;
    for (const auto& [id, points] : curves)
    {
        bool found = false;
        for (const auto& track : project.tracks)
            for (const auto& clip : track.clips)
                for (const auto& note : clip.notes) if (note.id == id) found = true;
        if (!found || points.empty()) return false;
        for (const auto& p : points)
            if (!std::isfinite(p.timeSeconds) || !std::isfinite(p.targetMidi)
                || p.targetMidi < 0 || p.targetMidi > 127) return false;
    }
    pushUndoLocked();
    const juce::ScopedValueSetter<bool> grouping(suppressNestedUndo, true);
    for (const auto& [id, points] : curves)
    {
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
                for (auto& note : clip.notes)
                    if (note.id == id) { note.vibratoEnabled = false; note.utauAutoPitchTransition = false; }
        setNotePitchCurve(id, points, true);
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
                for (auto& note : clip.notes)
                    if (note.id == id)
                    {
                        note.diffSingerPitchReference = note.pitchControlPoints;
                        note.diffSingerPitchReferenceFromSavedPitch = false;
                        note.pitchControlPoints = compactDiffSingerPitchHandles(note.pitchControlPoints,
                            note.diffSingerPitchReference, note.durationSeconds, preserveOutsideNoteHandles);
                    }
    }
    return true;
}

bool ProjectModel::setDiffSingerPitchOffset(const juce::String& noteId,
                                            std::vector<PitchCurveEditPoint> points)
{
    for (const auto& p : points)
        if (!std::isfinite(p.timeSeconds) || !std::isfinite(p.targetMidi)) return false;
    const juce::ScopedLock guard(lock);
    for (auto& track : project.tracks) if (trackIsDiffSinger(track))
        for (auto& clip : track.clips) for (auto& note : clip.notes) if (note.id == noteId)
        {
            for (auto& p : points)
            {
                p.timeSeconds = juce::jlimit(0.0, note.durationSeconds, p.timeSeconds);
                p.targetMidi = juce::jlimit(-127.0f, 127.0f, p.targetMidi);
                p.shape = PitchCurveShape::linear;
                p.diffSingerRestoreSupport = false;
            }
            std::stable_sort(points.begin(), points.end(), [](const auto& a, const auto& b){return a.timeSeconds<b.timeSeconds;});
            std::vector<PitchCurveEditPoint> clean;
            for (const auto& p : points)
                if (!clean.empty() && std::abs(p.timeSeconds-clean.back().timeSeconds)<1e-7) clean.back()=p;
                else clean.push_back(p);
            if (clean.size()<=2 && std::all_of(clean.begin(),clean.end(),[](const auto& p){return std::abs(p.targetMidi)<1e-6f;})) clean.clear();
            const auto& old=note.diffSingerPitchOffset;
            if (old.size()==clean.size() && std::equal(old.begin(),old.end(),clean.begin(),[](const auto& a,const auto& b){
                return a.timeSeconds==b.timeSeconds && a.targetMidi==b.targetMidi;})) return true;
            pushUndoLocked();note.diffSingerPitchOffset=std::move(clean);sendChangeMessage();return true;
        }
    return false;
}

bool ProjectModel::planPointCurvesLocked(const std::vector<PointCurveEdit>& edits,PointCurvePlanPreview& out,juce::String& error) const
{
    out={};error.clear();if(edits.empty()){error="No curve edits";return false;}
    std::set<const TrackData*> affected;
    const auto fail=[&](const juce::String& message){error=message;return false;};
    for(const auto& e:edits)
    {
        const bool graph=e.kind==PointCurveEdit::connectionPitch||e.kind==PointCurveEdit::connectionPitchStroke;
        if(graph)
        {
            if(std::count_if(project.nativeConnections.begin(),project.nativeConnections.end(),[&](const auto& c){return c.id==e.connectionId;})!=1)return fail("Unknown or ambiguous graph edit target");
            const auto* track=graphConnectionTrack(project,project.nativeConnections,e.connectionId,error);if(!track)return false;affected.insert(track);
            if(e.kind==PointCurveEdit::connectionPitch&&e.pitchPoints.size()<2)return fail("A graph curve needs at least two controls");
            for(const auto& p:e.pitchPoints)if(!finiteGraphPoint(p)||p.targetMidi<0||p.targetMidi>127)return fail("Invalid graph pitch control");
            continue;
        }
        const TrackData* owner=nullptr;const NoteData* note=nullptr;int count=0;
        for(const auto& t:project.tracks)for(const auto& c:t.clips)for(const auto& n:c.notes)if(n.id==e.noteId){owner=&t;note=&n;++count;}
        if(count!=1)return fail("Missing or ambiguous note edit target");affected.insert(owner);
        switch(e.kind)
        {
            case PointCurveEdit::pitch:
                if(note->nativeUnpitched||e.pitchPoints.empty())return fail("Missing pitched-note controls");
                for(const auto& p:e.pitchPoints)if(!finiteGraphPoint(p)||p.targetMidi<0||p.targetMidi>127
                    ||p.timeSeconds<(e.storePitchControlPoints?-30.0:0.0)||p.timeSeconds>note->durationSeconds+(e.storePitchControlPoints?30.0:0.0))return fail("Invalid note pitch controls");
                break;
            case PointCurveEdit::amplitude:
            {
                if(e.amplitudePoints.size()<2)return fail("An envelope needs at least two controls");
                std::vector<double> times;
                for(const auto& p:e.amplitudePoints)
                {if(!std::isfinite(p.timeSeconds)||!std::isfinite(p.gainDb)||p.timeSeconds< -5||p.timeSeconds>60||p.gainDb< -60||p.gainDb>12)return fail("Invalid amplitude controls");times.push_back(p.timeSeconds);}
                std::sort(times.begin(),times.end());int distinct=1;double previous=times.front();for(auto t:times)if(t>previous+1.e-5){++distinct;previous=t;}
                if(distinct<2)return fail("Envelope controls collapse to one time");break;
            }
            case PointCurveEdit::flag:
            {
                if(e.key.isEmpty()||e.key.startsWith("DS:REF:")||e.key.startsWith("DS:PTS:"))return fail("Read-only or missing parameter key");
                if(e.key.startsWith("DS:ABS:")&&!trackIsDiffSinger(*owner))return fail("This track has no DiffSinger parameter adapter");
                if(!e.flagPoints.empty()&&(!trackTakesFlagCurves(*owner)||trackIsDiffSinger(*owner)!=e.key.startsWith("DS:")))return fail("Unsupported parameter curve for this track");
                const auto& kind=flagCurveKindFor(e.key);
                for(const auto& p:e.flagPoints)if(!std::isfinite(p.timeSeconds)||!std::isfinite(p.value)||!std::isfinite(p.bezierX1)||!std::isfinite(p.bezierY1)
                    ||!std::isfinite(p.bezierX2)||!std::isfinite(p.bezierY2)||p.timeSeconds< -5||p.timeSeconds>60||p.value<kind.minimum||p.value>kind.maximum)return fail("Invalid parameter controls");
                break;
            }
            case PointCurveEdit::cvvc:
            {
                if(e.part<1||e.part>3||e.key.startsWith("DS:")||!owner->chineseCvvc||owner->utauMode!=UtauMode::classic||!trackUsesVoicebankSynthesis(*owner))return fail("Unsupported CVVC envelope target");
                backend::CvvcEnvelopeEdits input;input[e.part][e.key]=e.cvvcPoints;
                if(backend::cvvcEnvelopesFromVar(backend::cvvcEnvelopesToVar(input)).empty())return fail("Invalid CVVC envelope controls");
                for(const auto& p:e.cvvcPoints)if(!std::isfinite(p.position)||!std::isfinite(p.value))return fail("Invalid CVVC envelope value");
                break;
            }
            default:return fail("Unknown curve edit kind");
        }
    }
    ProjectModel draft;draft.suppressNestedUndo=true;draft.revision=revision;
    draft.project.bpm=project.bpm;draft.project.beatOriginSeconds=project.beatOriginSeconds;draft.project.tempoChanges=project.tempoChanges;
    draft.project.numerator=project.numerator;draft.project.denominator=project.denominator;draft.project.gridDivision=project.gridDivision;
    draft.project.noteEditDivision=project.noteEditDivision;draft.project.nativeConnections=project.nativeConnections;
    for(const auto& track:project.tracks)if(affected.count(&track))
    {
        if(track.id.isEmpty()||std::count_if(project.tracks.begin(),project.tracks.end(),[&](const auto& t){return t.id==track.id;})!=1)return fail("Missing or ambiguous track identity");
        draft.project.tracks.push_back(track); // real affected tracks, not a whole-project clone
    }
    const auto baseline=draft.contentFingerprint();
    // NOTE setters are the existing authorities, isolated with undo suppressed
    // and no listeners. Their source-fact/normalization behavior is not copied.
    for(const auto& e:edits)switch(e.kind)
    {
        case PointCurveEdit::pitch:if(!draft.setNotePitchCurve(e.noteId,e.pitchPoints,e.storePitchControlPoints,e.authorPitchSupport))return fail("Pitched note edit was rejected");break;
        case PointCurveEdit::amplitude:(void)draft.setNotesAmplitudeEnvelopes({{e.noteId,e.amplitudePoints}});break;
        case PointCurveEdit::flag:(void)draft.setNoteUtauFlagCurve(e.noteId,e.key,e.flagPoints);break;
        case PointCurveEdit::cvvc:(void)draft.setCvvcEnvelope(e.noteId,e.part,e.key,e.cvvcPoints,draft.revision);break;
        default:break;
    }
    // GRAPH commands see the proposed note score, never an old endpoint chord.
    for(const auto& e:edits)if(e.kind==PointCurveEdit::connectionPitch||e.kind==PointCurveEdit::connectionPitchStroke)
    {
        const auto* track=graphConnectionTrack(draft.project,draft.project.nativeConnections,e.connectionId,error);if(!track)return false;
        auto points=e.pitchPoints;
        if(e.kind==PointCurveEdit::connectionPitchStroke)
        {
            std::vector<PitchCurveEditPoint> composed;
            if(!composeConnectionPitchStroke(*track,draft.project.nativeConnections,e.connectionId,points,composed,error))
            {if(error.isNotEmpty())return false;continue;}
            points=std::move(composed);
        }
        else if(e.reanchorConnectionEndpoints)
        {
            const auto c=std::find_if(draft.project.nativeConnections.begin(),draft.project.nativeConnections.end(),[&](const auto& c){return c.id==e.connectionId;});
            const auto endpoints=graphPitchEndpoints(*track);const auto a=std::find_if(endpoints.begin(),endpoints.end(),[&](const auto& n){return n.note->id==c->leftNoteId;});
            const auto b=std::find_if(endpoints.begin(),endpoints.end(),[&](const auto& n){return n.note->id==c->rightNoteId;});
            if(a==endpoints.end()||b==endpoints.end())return fail("Missing proposed graph endpoint");
            const auto base=sharedPitchLines(*track);
            points.front().targetMidi=graphBaseMidiAt(base,*a,points.front().timeSeconds,{},nullptr);
            points.back().targetMidi=graphBaseMidiAt(base,*b,points.back().timeSeconds,{},nullptr);
        }
        (void)prepareConnectionPitchEdit(*track,draft.project.nativeConnections,e.connectionId,points,error);if(error.isNotEmpty())return false;
    }
    out.changed=draft.contentFingerprint()!=baseline;out.tracks=std::move(draft.project.tracks);out.connections=std::move(draft.project.nativeConnections);return true;
}

bool ProjectModel::previewPointCurves(const std::vector<PointCurveEdit>& edits,std::uint64_t expectedRevision,
    PointCurvePlanPreview& out,juce::String& error) const
{
    const juce::ScopedLock guard(lock);out={};error.clear();
    if(revision!=expectedRevision){error="Project changed during the curve gesture";return false;}
    return planPointCurvesLocked(edits,out,error);
}

bool ProjectModel::setPointCurves(std::vector<PointCurveEdit> edits, std::uint64_t expectedRevision,juce::String* diagnostic)
{
    const juce::ScopedLock guard(lock);if(diagnostic)diagnostic->clear();
    if(revision!=expectedRevision||edits.empty()){if(diagnostic)*diagnostic="Stale or empty curve gesture";return false;}
    PointCurvePlanPreview planned;juce::String error;
    if(!planPointCurvesLocked(edits,planned,error)){if(diagnostic)*diagnostic=error;return false;}
    if(!planned.changed)return false;
    pushUndoLocked();
    for(auto& track:planned.tracks)
    {
        const auto at=std::find_if(project.tracks.begin(),project.tracks.end(),[&](const auto& t){return t.id==track.id;});
        *at=std::move(track);
    }
    project.nativeConnections=std::move(planned.connections);sendChangeMessage();return true;
}

bool ProjectModel::setNotePitchCurve(const juce::String& noteId,
                                     std::vector<PitchCurveEditPoint> points,
                                     bool storeControlPoints, bool authorPitchSupport)
{
    if (points.empty()) return false;
    std::stable_sort(points.begin(), points.end(), [](const auto& left, const auto& right)
    {
        return left.timeSeconds < right.timeSeconds;
    });
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
                for (auto& note : clip.notes)
                {
                    if (note.id != noteId || note.nativeUnpitched) continue;
                    for (auto& point : points)
                    {
                        // Handles reach before the note, and a UST bend runs
                        // on past its end.  Held to the note's own span, every
                        // point past the end was pulled back to it whenever any
                        // one point of the note was touched -- moving points
                        // nobody had moved, and reshaping the note's tail.
                        const auto minimumTime = storeControlPoints ? -30.0 : 0.0;
                        const auto maximumTime = storeControlPoints
                            ? note.durationSeconds + 30.0 : note.durationSeconds;
                        point.timeSeconds = juce::jlimit(minimumTime, maximumTime,
                                                         point.timeSeconds);
                        point.targetMidi = juce::jlimit(0.0f, 127.0f, point.targetMidi);
                    }
                    pushUndoLocked();
                    changed = true;
                    if (trackShowsAllNativeRegions(track)) note.nativePitchHandlesPlaced = true;
                    if(authorPitchSupport)note.pitchSupportAuthored=true; // explicit OR; local edits preserve prior intent
                    if (storeControlPoints) note.pitchControlPoints = points;
                    else note.pitchControlPoints.clear();

                    // User-created notes initially contain only two endpoints.
                    // Densify them before drawing so a freehand edit has the
                    // same 5 ms precision as imported Melodyne/FCPE contours.
                    auto needsDensifying = note.contour.size() < 2;
                    for (std::size_t index = 1; index < note.contour.size(); ++index)
                        needsDensifying = needsDensifying
                            || note.contour[index].timeSeconds
                                - note.contour[index - 1].timeSeconds > 0.0075;
                    if(trackShowsAllNativeRegions(track))
                    {
                        // A local native stroke must not rebuild/erase factual
                        // source knots elsewhere. Union interpolated supports,
                        // retaining every old tuple and its voiced/manual flags.
                        const auto original=note.contour;
                        const auto sourceAt=[&](double t)
                        {
                            PitchPoint p;p.timeSeconds=t;if(original.empty())return p;
                            auto r=std::lower_bound(original.begin(),original.end(),t,[](const auto& p,double v){return p.timeSeconds<v;});
                            if(r==original.end()){p=original.back();p.timeSeconds=t;return p;}
                            if(r==original.begin()||r->timeSeconds==t){p=*r;p.timeSeconds=t;return p;}
                            const auto& a=*std::prev(r);p=a;const auto u=static_cast<float>((t-a.timeSeconds)/(r->timeSeconds-a.timeSeconds));
                            p.timeSeconds=t;p.relativeCents+=(r->relativeCents-a.relativeCents)*u;p.withoutVibratoCents+=(r->withoutVibratoCents-a.withoutVibratoCents)*u;
                            p.voiced=a.voiced&&r->voiced;p.hasManualTarget=a.hasManualTarget&&r->hasManualTarget;
                            if(p.hasManualTarget)p.manualTargetCents+=(r->manualTargetCents-a.manualTargetCents)*u;return p;
                        };
                        std::vector<double> supports;
                        const auto from=storeControlPoints?0.0:points.front().timeSeconds;
                        const auto to=storeControlPoints?note.durationSeconds:points.back().timeSeconds;
                        if(needsDensifying)for(double t=from;t<to;t+=.005)supports.push_back(t);
                        for(const auto& p:points)if(p.timeSeconds>=0&&p.timeSeconds<=note.durationSeconds)supports.push_back(p.timeSeconds);
                        if(original.size()<2){supports.push_back(0);supports.push_back(note.durationSeconds);}
                        std::sort(supports.begin(),supports.end());supports.erase(std::unique(supports.begin(),supports.end()),supports.end());
                        for(auto t:supports)
                        {
                            const auto r=std::lower_bound(original.begin(),original.end(),t,[](const auto& p,double v){return p.timeSeconds<v;});
                            if(r==original.end()||r->timeSeconds!=t)note.contour.push_back(sourceAt(t));
                        }
                        std::stable_sort(note.contour.begin(),note.contour.end(),[](const auto& a,const auto& b){return a.timeSeconds<b.timeSeconds;});
                    }
                    else if (needsDensifying)
                    {
                        const auto original = note.contour;
                        const auto evaluate = [&](double time)
                        {
                            PitchPoint result;
                            result.timeSeconds = time;
                            if (original.empty()) return result;
                            const auto right = std::lower_bound(original.begin(), original.end(), time,
                                [](const PitchPoint& point, double value)
                                {
                                    return point.timeSeconds < value;
                                });
                            const auto rightIndex = static_cast<std::size_t>(right == original.end()
                                ? original.size() - 1 : right - original.begin());
                            const auto leftIndex = rightIndex > 0
                                && original[rightIndex].timeSeconds > time ? rightIndex - 1 : rightIndex;
                            const auto& left = original[leftIndex];
                            const auto& next = original[rightIndex];
                            const auto amount = next.timeSeconds > left.timeSeconds
                                ? static_cast<float>(juce::jlimit(0.0, 1.0,
                                    (time - left.timeSeconds)
                                        / (next.timeSeconds - left.timeSeconds))) : 0.0f;
                            result.relativeCents = left.relativeCents
                                + (next.relativeCents - left.relativeCents) * amount;
                            result.withoutVibratoCents = left.withoutVibratoCents
                                + (next.withoutVibratoCents - left.withoutVibratoCents) * amount;
                            result.voiced = left.voiced && next.voiced;
                            if (left.hasManualTarget && next.hasManualTarget)
                            {
                                result.hasManualTarget = true;
                                result.manualTargetCents = left.manualTargetCents
                                    + (next.manualTargetCents - left.manualTargetCents) * amount;
                            }
                            return result;
                        };
                        note.contour.clear();
                        for (double time = 0.0; time < note.durationSeconds; time += 0.005)
                            note.contour.push_back(evaluate(time));
                        note.contour.push_back(evaluate(note.durationSeconds));
                    }

                    const auto firstTime = points.front().timeSeconds;
                    const auto lastTime = points.back().timeSeconds;
                    const auto targetAt = [&](double time)
                    {
                        return evaluatePitchCurve(points, time);
                    };
                    if (points.size() == 1 && !note.contour.empty())
                    {
                        auto nearest = std::min_element(note.contour.begin(), note.contour.end(),
                            [&](const auto& left, const auto& right)
                            {
                                return std::abs(left.timeSeconds - firstTime)
                                    < std::abs(right.timeSeconds - firstTime);
                            });
                        nearest->manualTargetCents =
                            (points.front().targetMidi - note.midiNote) * 100.0f;
                        nearest->hasManualTarget = true;
                    }
                    else
                    {
                        // Control points describe the whole note, so they apply
                        // to all of it.  Anchors get trimmed away from the ends
                        // where a note abuts its neighbour, and writing only
                        // between the outermost anchors left the contour holding
                        // its old value beyond them - a step down at the first
                        // anchor and back up at the last, which is what showed
                        // up as a break across the consonant even though the
                        // anchors themselves were continuous.  Outside the
                        // anchor span evaluatePitchCurve holds the end value,
                        // matching what encodePitchbend sends.
                        //
                        // A freehand or line stroke is a local edit and stays
                        // confined to the stretch it covers, so that it cannot
                        // wipe measured pitch elsewhere in the note.
                        // Native UV samples retain the authored guide too, but
                        // keep voiced=false: the guide never creates source F0.
                        for (auto& point : note.contour)
                            if ((point.voiced || trackShowsAllNativeRegions(track))
                                && (storeControlPoints
                                    || (point.timeSeconds >= firstTime - 1.0e-7
                                        && point.timeSeconds <= lastTime + 1.0e-7)))
                            {
                                point.manualTargetCents =
                                    (targetAt(point.timeSeconds) - note.midiNote) * 100.0f;
                                point.hasManualTarget = true;
                            }
                    }
                    break;
                }
    }
    if (changed) sendChangeMessage();
    return changed;
}

ProjectModel::PlannedNote ProjectModel::plannedNoteFor(
    double startSeconds, double requestedSeconds, double clipSeconds,
    const std::vector<NoteSpan>& occupied)
{
    // Anything shorter than this is widened to the model's own floor below,
    // which would put the overlap straight back in.
    constexpr auto shortest = 0.01;
    constexpr auto epsilon = 1.0e-9;

    PlannedNote planned;
    if (startSeconds < -epsilon) return planned;
    // Past the end of the clip there is no room.  Clamping instead is what
    // used to stack notes on the final instant.
    if (startSeconds > clipSeconds - shortest + epsilon) return planned;

    auto room = std::min(requestedSeconds, clipSeconds - startSeconds);
    for (const auto& span : occupied)
    {
        const auto end = span.startSeconds + span.durationSeconds;
        // The moment asked for is inside a note that is already there.
        if (span.startSeconds <= startSeconds + epsilon
            && startSeconds < end - epsilon)
            return planned;
        // A note begins further along: the new one stops where it starts.
        if (span.startSeconds > startSeconds)
            room = std::min(room, span.startSeconds - startSeconds);
    }
    if (room < shortest - epsilon) return planned;

    planned.create = true;
    planned.startSeconds = startSeconds;
    planned.durationSeconds = room;
    return planned;
}

namespace
{
// How far into a clip a note may reach.  A clip with a recording behind it is
// as long as its audio and that is the end of it; one composed here is only a
// span of the timeline, so a note may push it out.
double writableEndOf(const ClipData& clip, double startSeconds, double duration)
{
    if (clip.sourceFile != juce::File()) return clip.durationSeconds;
    return std::max(clip.durationSeconds, startSeconds + std::max(0.0, duration));
}
}

bool ProjectModel::canAddEmptyTuningClip(const juce::String& trackId, double startSeconds) const
{
    if (!std::isfinite(startSeconds) || startSeconds < 0.0) return false;
    const juce::ScopedLock guard(lock);
    for (const auto& track : project.tracks)
        if (track.id == trackId && !track.accompaniment)
            return std::none_of(track.clips.begin(),track.clips.end(),[&](const auto& clip)
            { return startSeconds >= clip.startSeconds && startSeconds < clip.startSeconds+clip.durationSeconds; });
    return false;
}

juce::String ProjectModel::addEmptyTuningClip(const juce::String& trackId, double startSeconds, double durationSeconds)
{
    if (!std::isfinite(durationSeconds) || durationSeconds < 0.01 || !std::isfinite(startSeconds+durationSeconds)) return {};
    juce::String created;
    {
        const juce::ScopedLock guard(lock);
        if (!canAddEmptyTuningClip(trackId,startSeconds)) return {};
        for (auto& track : project.tracks) if (track.id == trackId)
        {
            pushUndoLocked();
            track.compose = true;
            ClipData clip;clip.id=created=makeId("clip");clip.startSeconds=startSeconds;
            clip.durationSeconds=clip.sourceDurationSeconds=durationSeconds;
            track.clips.push_back(std::move(clip));
            std::stable_sort(track.clips.begin(),track.clips.end(),[](const auto& a,const auto& b){return a.startSeconds<b.startSeconds;});
            break;
        }
    }
    if (created.isNotEmpty()) sendChangeMessage();
    return created;
}

juce::String ProjectModel::addClip(const juce::String& trackId,
                                   double startSeconds, double durationSeconds)
{
    juce::String created;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            if (track.id == trackId && track.compose)
            {
                pushUndoLocked();
                ClipData clip;
                clip.id = makeId("clip");
                clip.startSeconds = std::max(0.0, startSeconds);
                clip.durationSeconds = std::max(0.01, durationSeconds);
                clip.sourceDurationSeconds = clip.durationSeconds;
                created = clip.id;
                track.clips.push_back(std::move(clip));
                std::stable_sort(track.clips.begin(), track.clips.end(),
                    [](const auto& left, const auto& right)
                    { return left.startSeconds < right.startSeconds; });
                break;
            }
    }
    if (created.isNotEmpty()) sendChangeMessage();
    return created;
}

juce::String ProjectModel::addNoteLocked(ClipData& destination,
                                         const PlannedNote& planned, float midiNote)
{
    pushUndoLocked();
    NoteData note;
    for (const auto& track : project.tracks) for (const auto& clip : track.clips)
        if (&clip == &destination) note.utauFlagCurveEnabled = trackIsDiffSinger(track);
    note.id = makeId("note");
    note.startSeconds = planned.startSeconds;
    note.durationSeconds = planned.durationSeconds;
    note.consonantSeconds = std::min(0.04, note.durationSeconds * 0.3);
    note.midiNote = juce::jlimit(0.0f, 127.0f, midiNote);
    note.sourceMidiCenter = note.midiNote;
    note.contour.push_back({ 0.0, 0.0f, 0.0f, true });
    note.contour.push_back({ note.durationSeconds, 0.0f, 0.0f, true });
    const auto id = note.id;
    const auto reach = note.startSeconds + note.durationSeconds;
    destination.notes.push_back(std::move(note));
    if (destination.sourceFile == juce::File() && destination.durationSeconds < reach)
    {
        destination.durationSeconds = reach;
        destination.sourceDurationSeconds = reach;
    }
    std::stable_sort(destination.notes.begin(), destination.notes.end(),
        [](const auto& left, const auto& right) { return left.startSeconds < right.startSeconds; });
    return id;
}

juce::String ProjectModel::insertPrefixNote(const juce::String& noteId,
                                           double targetOverlapSeconds)
{
    juce::String created;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
            {
                const auto found = std::find_if(clip.notes.begin(), clip.notes.end(),
                    [&noteId](const auto& note) { return note.id == noteId; });
                if (found == clip.notes.end()) continue;
                if (found->startSeconds <= 1.0e-9) return {};
                pushUndoLocked();
                // Its vowel from its own beat, so the consonant is heard up to
                // it.  The overlap it had goes back in with the pin, since the
                // two are pinned together.
                found->utauPreutteranceOverrideEnabled = true;
                found->utauPreutteranceSeconds = 0.0;
                found->utauOverlapOverrideEnabled = true;
                found->utauOverlapSeconds = std::isfinite(targetOverlapSeconds)
                    ? targetOverlapSeconds : 0.0;
                NoteData prefix;
                prefix.id = makeId("note");
                prefix.startSeconds = found->startSeconds;
                prefix.durationSeconds = 0.0;
                prefix.consonantSeconds = 0.0;
                prefix.midiNote = found->midiNote;
                prefix.sourceMidiCenter = found->midiNote;
                prefix.label = found->label;
                prefix.contour.push_back({ 0.0, 0.0f, 0.0f, true });
                prefix.contour.push_back({ 0.0, 0.0f, 0.0f, true });
                created = prefix.id;
                clip.notes.insert(found, std::move(prefix));
                break;
            }
    }
    if (created.isNotEmpty()) sendChangeMessage();
    return created;
}

void ProjectModel::flattenNotePitch(const std::vector<juce::String>& noteIds, bool snapNativeToSemitone)
{
    if (noteIds.empty()) return;
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
            for (auto& clip : track.clips)
                for (auto& note : clip.notes)
                {
                    if (std::find(noteIds.begin(), noteIds.end(), note.id)
                        == noteIds.end())
                        continue;
                    if (note.nativeUnpitched) continue;
                    if (!changed) pushUndoLocked();
                    changed = true;

                    if (snapNativeToSemitone && trackShowsAllNativeRegions(track))
                        note.midiNote = juce::jlimit(0.0f, 127.0f, std::round(note.midiNote));
                    if (trackShowsAllNativeRegions(track))
                    {
                        note.nativeIndependentPitch = true;
                        note.nativePitchHandlesPlaced = false;
                        note.utauAutoPitchTransition = false;
                        note.vibratoEnabled = false;
                    }
                    for (auto& point : note.contour)
                    {
                        // Every frame, not only the voiced ones.  Where nothing
                        // was detected the note should move with the rest
                        // rather than keep whatever it was holding.
                        point.manualTargetCents = 0.0f;
                        point.hasManualTarget = true;
                    }
                    // Drift and modulation reshape the measured curve, and
                    // there is no longer a curve to reshape.
                    note.drift = 0.0f;
                    note.modulation = 0.0f;
                    note.pitchControlPoints = {
                        { 0.0, note.midiNote },
                        { note.durationSeconds, note.midiNote },
                    };
                }
    }
    if (changed) sendChangeMessage();
}

bool ProjectModel::restoreNativeSourcePitch(const std::vector<juce::String>& noteIds)
{
    bool changed = false;
    {
        const juce::ScopedLock guard(lock);
        std::vector<NoteData*> targets;
        std::vector<juce::String> restoredIds;
        for (auto& track : project.tracks)
        {
            if (!trackShowsAllNativeRegions(track)) continue;
            for (auto& clip : track.clips) for (auto& note : clip.notes)
                if (std::find(noteIds.begin(), noteIds.end(), note.id) != noteIds.end()
                    && !note.nativeUnpitched && std::isfinite(note.sourceMidiCenter) && note.sourceMidiCenter >= 0.0f
                    && !note.contour.empty())
                { targets.push_back(&note); restoredIds.push_back(note.id); }
        }
        if (targets.empty()) return false;
        const auto includes = [&](const juce::String& id) {
            return std::find(restoredIds.begin(), restoredIds.end(), id) != restoredIds.end();
        };
        const auto edited = [](const NoteData* note) {
            return note->midiNote != note->sourceMidiCenter || note->drift != 1.0f
                || note->modulation != 1.0f || !note->pitchControlPoints.empty()
                || note->vibratoEnabled || note->vibratoRealLine || note->robustPitchCurve
                || note->connectedToPrevious || note->connectedToNext || note->utauAutoPitchTransition
                || std::any_of(note->contour.begin(), note->contour.end(),
                    [](const auto& point) { return point.hasManualTarget; });
        };
        changed = std::any_of(targets.begin(), targets.end(), edited)
            || std::any_of(project.nativeConnections.begin(), project.nativeConnections.end(),
                [&](const auto& connection) { return includes(connection.leftNoteId) || includes(connection.rightNoteId); });
        if (!changed) return false;
        pushUndoLocked();
        // Unlink pitch glides at both endpoints, including older projects with
        // only compatibility flags.  A synthetic glide must not reshape restored F0.
        for (auto& track : project.tracks)
        {
            if (!trackShowsAllNativeRegions(track)) continue;
            struct Placed { NoteData* note; ClipData* clip; double start; };
            std::vector<Placed> ordered;
            for (auto& clip : track.clips) for (auto& note : clip.notes)
                ordered.push_back({&note, &clip, clip.startSeconds + note.startSeconds});
            std::stable_sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) { return a.start < b.start; });
            for (std::size_t i = 0; i < ordered.size(); ++i)
            {
                if (!includes(ordered[i].note->id)) continue;
                auto& note = *ordered[i].note;
                if (note.connectedToPrevious && i > 0)
                { ordered[i-1].note->connectedToNext = false; ordered[i-1].clip->glideConnectedToNext = false; }
                if (note.connectedToNext && i+1 < ordered.size())
                { ordered[i+1].note->connectedToPrevious = false; ordered[i+1].clip->glideConnectedFromPrevious = false; }
                if (note.connectedToPrevious) ordered[i].clip->glideConnectedFromPrevious = false;
                if (note.connectedToNext) ordered[i].clip->glideConnectedToNext = false;
            }
        }
        for (auto& track : project.tracks) for (auto& clip : track.clips) for (auto& note : clip.notes)
            for (const auto& connection : project.nativeConnections)
                if (includes(connection.leftNoteId) || includes(connection.rightNoteId))
                {
                    if (note.id == connection.leftNoteId) { note.connectedToNext = false; clip.glideConnectedToNext = false; }
                    if (note.id == connection.rightNoteId) { note.connectedToPrevious = false; clip.glideConnectedFromPrevious = false; }
                }
        std::erase_if(project.nativeConnections, [&](const auto& connection) {
            return includes(connection.leftNoteId) || includes(connection.rightNoteId);
        });
        for (auto* note : targets)
        {
            note->midiNote = note->sourceMidiCenter;
            note->drift = note->modulation = 1.0f;
            note->pitchControlPoints.clear();
            note->materialTargetPitchLocked=false;
            note->pitchSupportAuthored=false; // explicit restore-source action removes authored target support
            note->vibratoEnabled = note->vibratoRealLine = false;
            note->robustPitchCurve = false;
            note->connectedToPrevious = note->connectedToNext = false;
            note->utauAutoPitchTransition = false;
            for (auto& point : note->contour)
            { point.hasManualTarget = false; point.manualTargetCents = 0.0f; }
        }
    }
    if (changed) sendChangeMessage();
    return changed;
}

double ProjectModel::firstFreeStartFrom(double startSeconds,
                                        const std::vector<NoteSpan>& occupied)
{
    constexpr auto epsilon = 1.0e-9;
    auto start = startSeconds;
    // One pass per note at most: each step lands on the end of a note that
    // covered the last position, and a note cannot be crossed twice.  The
    // bound also keeps a zero-length note from spinning here.
    for (std::size_t step = 0; step <= occupied.size(); ++step)
    {
        auto moved = false;
        for (const auto& span : occupied)
        {
            const auto end = span.startSeconds + span.durationSeconds;
            if (span.startSeconds <= start + epsilon && start < end - epsilon)
            {
                start = end;
                moved = true;
            }
        }
        if (!moved) break;
    }
    return start;
}

juce::String ProjectModel::addNoteFrom(const juce::String& preferredClipId,
                                       double absoluteFromSeconds,
                                       double maximumDuration, float midiNote)
{
    juce::String created;
    {
        const juce::ScopedLock guard(lock);
        ClipData* destination = nullptr;
        for (auto& track : project.tracks)
            if (track.compose)
                for (auto& clip : track.clips)
                {
                    if (clip.id == preferredClipId) destination = &clip;
                    if (destination == nullptr
                        && absoluteFromSeconds >= clip.startSeconds
                        && absoluteFromSeconds <= clip.startSeconds + clip.durationSeconds)
                        destination = &clip;
                }
        if (destination != nullptr)
        {
            std::vector<NoteSpan> occupied;
            occupied.reserve(destination->notes.size());
            for (const auto& existing : destination->notes)
                occupied.push_back({ existing.startSeconds, existing.durationSeconds });
            const auto from = firstFreeStartFrom(
                absoluteFromSeconds - destination->startSeconds, occupied);
            const auto planned = plannedNoteFor(from, maximumDuration,
                                                writableEndOf(*destination, from,
                                                              maximumDuration),
                                                occupied);
            if (planned.create)
                created = addNoteLocked(*destination, planned, midiNote);
        }
    }
    if (created.isNotEmpty()) sendChangeMessage();
    return created;
}

juce::String ProjectModel::addNote(const juce::String& preferredClipId,
                                   double absoluteStart, double duration, float midiNote)
{
    juce::String created;
    {
        const juce::ScopedLock guard(lock);
        ClipData* destination = nullptr;
        for (auto& track : project.tracks)
            if (track.compose)
                for (auto& clip : track.clips)
                {
                    if (clip.id == preferredClipId) destination = &clip;
                    if (destination == nullptr
                        && absoluteStart >= clip.startSeconds
                        && absoluteStart <= clip.startSeconds + clip.durationSeconds)
                        destination = &clip;
                }
        if (destination != nullptr)
        {
            std::vector<NoteSpan> occupied;
            occupied.reserve(destination->notes.size());
            for (const auto& existing : destination->notes)
                occupied.push_back({ existing.startSeconds, existing.durationSeconds });
            const auto local = absoluteStart - destination->startSeconds;
            const auto planned = plannedNoteFor(local, duration,
                                                writableEndOf(*destination, local, duration),
                                                occupied);
            if (!planned.create) return {};
            created = addNoteLocked(*destination, planned, midiNote);
        }
    }
    if (created.isNotEmpty()) sendChangeMessage();
    return created;
}

void ProjectModel::removeNote(const juce::String& noteId)
{
    removeNotes({ noteId });
}

std::vector<std::size_t> ProjectModel::regionsToImport(
    const std::vector<RegionSpan>& regions)
{
    constexpr auto epsilon = 1.0e-9;
    // Longest first, so the row that covers the most of the syllable is the
    // one kept; ties go to whichever came first in the file, which keeps the
    // result stable rather than dependent on the sort.
    std::vector<std::size_t> order(regions.size());
    for (std::size_t index = 0; index < order.size(); ++index) order[index] = index;
    std::stable_sort(order.begin(), order.end(),
        [&regions](std::size_t left, std::size_t right)
        {
            const auto leftLength = regions[left].endSeconds - regions[left].startSeconds;
            const auto rightLength = regions[right].endSeconds - regions[right].startSeconds;
            // Two rows of the same span at different offsets do not subtract
            // to bitwise equal lengths -- 0.3-0.1 and 0.4-0.2 differ in the
            // last bits -- and which alias you get must not hinge on that.
            // Equal within a nanosecond counts as equal, and stable_sort then
            // leaves the earlier row first.
            if (std::abs(leftLength - rightLength) <= epsilon) return false;
            return leftLength > rightLength;
        });

    std::vector<std::size_t> kept;
    for (const auto candidate : order)
    {
        const auto& region = regions[candidate];
        if (region.endSeconds - region.startSeconds < epsilon) continue;
        const auto clashes = std::any_of(kept.begin(), kept.end(),
            [&](std::size_t taken)
            {
                return region.startSeconds < regions[taken].endSeconds - epsilon
                    && regions[taken].startSeconds < region.endSeconds - epsilon;
            });
        if (!clashes) kept.push_back(candidate);
    }
    // Back into the order the file gave them, so the notes read left to right.
    std::sort(kept.begin(), kept.end());
    return kept;
}

bool ProjectModel::clipIsRecording(const ClipData& clip)
{
    if(!clip.parts.empty())return std::any_of(clip.parts.begin(),clip.parts.end(),[](const auto& p){return clipIsRecording(p);});
    if (clip.sourceFile == juce::File()) return false;
    return !clip.sourceFile.hasFileExtension("ust;mid;midi;mpd;hjpx;hspx");
}

void ProjectModel::dropEmptyRecordedClipsLocked()
{
    for (auto& track : project.tracks)
    {
        for(auto& clip:track.clips)std::erase_if(clip.parts,[&](const auto& part)
        {
            return part.materialSourceFingerprint.isNotEmpty()
                &&std::none_of(clip.notes.begin(),clip.notes.end(),[&](const auto& n){return n.clipPartId==part.id;});
        });
        std::erase_if(track.clips, [](const ClipData& clip)
        {
            return clip.notes.empty() && clipIsRecording(clip);
        });
    }
}

void ProjectModel::removeNotes(const std::vector<juce::String>& noteIds)
{
    const auto includes = [&](const juce::String& id)
    {
        return std::find(noteIds.begin(), noteIds.end(), id) != noteIds.end();
    };
    auto changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (const auto& track : project.tracks)
            for (const auto& clip : track.clips)
                for (const auto& note : clip.notes)
                    changed = changed || includes(note.id);
        if (!changed) return;
        pushUndoLocked();
        for (auto& track : project.tracks)
        {
            if(trackShowsAllNativeRegions(track))
                for(auto& clip:track.clips)rememberNativeTrimSources(clip);
            struct Positioned { NoteData* note; double start; };
            std::vector<Positioned> ordered;
            for (auto& clip : track.clips)
                for (auto& note : clip.notes)
                    ordered.push_back({ &note, clip.startSeconds + note.startSeconds });
            std::stable_sort(ordered.begin(), ordered.end(),
                [](const auto& left, const auto& right) { return left.start < right.start; });
            for (std::size_t index = 0; index < ordered.size(); ++index)
                if (includes(ordered[index].note->id))
                {
                    if (index > 0 && !includes(ordered[index - 1].note->id))
                        ordered[index - 1].note->connectedToNext = false;
                    if (index + 1 < ordered.size() && !includes(ordered[index + 1].note->id))
                        ordered[index + 1].note->connectedToPrevious = false;
                }
            if (trackShowsAllNativeRegions(track))
            {
                std::vector<ClipData> retained;
                for (const auto& original : track.clips)
                {
                    auto parts = removeNativeUnpitchedAudio(original, noteIds);
                    for (std::size_t i = 0; i < parts.size(); ++i)
                    {
                        parts[i].id = i == 0 ? original.id : makeId("clip");
                        retained.push_back(std::move(parts[i]));
                    }
                }
                track.clips = std::move(retained);
            }
            for (auto& clip : track.clips)
                clip.notes.erase(std::remove_if(clip.notes.begin(), clip.notes.end(),
                    [&](const auto& note) { return includes(note.id); }), clip.notes.end());
        }
        std::erase_if(project.nativeConnections, [&](const auto& c)
        { return includes(c.leftNoteId) || includes(c.rightNoteId); });
        dropEmptyRecordedClipsLocked();
    }
    if (changed) sendChangeMessage();
}

namespace
{
struct ConnectionEndpoint
{
    NoteData* note;
    ClipData* clip;
    double start;
};

std::vector<ConnectionEndpoint> orderedConnectionEndpoints(TrackData& track)
{
    std::vector<ConnectionEndpoint> ordered;
    for (auto& clip : track.clips)
        for (auto& note : clip.notes)
            ordered.push_back({ &note, &clip, clip.startSeconds + note.startSeconds });
    std::stable_sort(ordered.begin(), ordered.end(),
        [](const auto& left, const auto& right) { return left.start < right.start; });
    return ordered;
}

bool hasConnectionPair(const std::vector<NativeConnection>& connections,
                       const ConnectionEndpoint& left, const ConnectionEndpoint& right)
{
    return std::any_of(connections.begin(), connections.end(), [&](const auto& connection)
    { return connection.leftNoteId == left.note->id && connection.rightNoteId == right.note->id; });
}

bool validNewConnectionPair(const ConnectionEndpoint& left, const ConnectionEndpoint& right)
{
    // A zero-duration source lead-in is legitimate. Independent overlapping
    // bodies are not a boundary and must not acquire a join by start order alone.
    return left.note != right.note && left.note->id.isNotEmpty() && right.note->id.isNotEmpty()
        && left.note->id != right.note->id && std::isfinite(left.start) && std::isfinite(right.start)
        && std::isfinite(left.note->durationSeconds) && left.note->durationSeconds >= 0.0
        && std::isfinite(right.note->durationSeconds) && right.note->durationSeconds >= 0.0
        && left.start + left.note->durationSeconds <= right.start + 1.0e-7;
}

void appendNativeConnection(std::vector<NativeConnection>& connections,
                            const ConnectionEndpoint& left, const ConnectionEndpoint& right)
{
    connections.push_back({ "connection_" + juce::Uuid().toString().removeCharacters("-"),
        left.note->id, right.note->id, "pitch-and-amplitude", (left.start + right.start) * 0.5, {}, {} });
}

bool materializeLegacyConnectionPairs(std::vector<NativeConnection>& connections,
                                     const std::vector<ConnectionEndpoint>& ordered)
{
    bool changed = false;
    for (std::size_t i = 1; i < ordered.size(); ++i)
    {
        const auto& left = ordered[i - 1]; const auto& right = ordered[i];
        const bool reciprocalNoteJoin = left.note->connectedToNext && right.note->connectedToPrevious;
        // Some importers/renderers encode the legacy join on just one side.
        // Recover that intent only at a non-overlapping adjacent boundary; a
        // reciprocal existing join is retained even when its bodies overlap.
        const bool noteJoin = (left.note->connectedToNext || right.note->connectedToPrevious)
            && (reciprocalNoteJoin || validNewConnectionPair(left, right));
        const bool clipJoin = left.clip != right.clip && (left.clip->glideConnectedToNext
            || right.clip->glideConnectedFromPrevious) && validNewConnectionPair(left, right);
        if ((!noteJoin && !clipJoin) || left.note->id.isEmpty() || right.note->id.isEmpty()
            || left.note->id == right.note->id || hasConnectionPair(connections, left, right)) continue;
        // Explicit directional endpoints are authoritative. Their compatibility
        // flags must not manufacture a second, guessed adjacent partner.
        if (std::any_of(connections.begin(), connections.end(), [&](const auto& connection)
            { return connection.leftNoteId == left.note->id || connection.rightNoteId == right.note->id; })) continue;
        appendNativeConnection(connections, left, right);
        changed = true;
    }
    return changed;
}

bool mutateNativeConnectionPair(std::vector<NativeConnection>& connections,
                                const ConnectionEndpoint& left, const ConnectionEndpoint& right,
                                bool enabled)
{
    const auto count = connections.size();
    const bool existing = hasConnectionPair(connections, left, right);
    std::erase_if(connections, [&](const auto& connection)
    {
        const bool exact = connection.leftNoteId == left.note->id && connection.rightNoteId == right.note->id;
        // Adding B-C can replace B's outgoing edge and C's incoming edge, but
        // never A-B or C-D. Removing a boundary removes only that exact pair.
        return enabled ? !exact && (connection.leftNoteId == left.note->id || connection.rightNoteId == right.note->id)
                       : exact;
    });
    const bool erased = count != connections.size();
    if (enabled && !existing) appendNativeConnection(connections, left, right);
    return erased || (enabled && !existing);
}

bool projectConnectionFlags(const std::vector<NativeConnection>& connections,
                            const std::vector<ConnectionEndpoint>& ordered, bool apply)
{
    std::map<juce::String, const ConnectionEndpoint*> byId;
    std::map<NoteData*, std::pair<bool, bool>> noteFlags;
    std::map<ClipData*, std::pair<bool, bool>> clipFlags;
    for (const auto& endpoint : ordered)
    {
        byId.emplace(endpoint.note->id, &endpoint);
        noteFlags.emplace(endpoint.note, std::pair{ false, false });
        clipFlags.emplace(endpoint.clip, std::pair{ false, false });
    }
    for (const auto& connection : connections)
    {
        if(connection.type=="pitch-portamento")continue; // pitch only: never concatenate/fade sources
        const auto left = byId.find(connection.leftNoteId), right = byId.find(connection.rightNoteId);
        // Unknown, self and cross-track records cannot project flags here.
        if (left == byId.end() || right == byId.end() || left == right) continue;
        noteFlags[left->second->note].second = true;
        noteFlags[right->second->note].first = true;
        if (left->second->clip != right->second->clip)
        {
            clipFlags[left->second->clip].second = true;
            clipFlags[right->second->clip].first = true;
        }
    }
    bool changed = false;
    for (const auto& [note, flags] : noteFlags)
    {
        changed |= note->connectedToPrevious != flags.first || note->connectedToNext != flags.second;
        if (apply) { note->connectedToPrevious = flags.first; note->connectedToNext = flags.second; }
    }
    for (const auto& [clip, flags] : clipFlags)
    {
        changed |= clip->glideConnectedFromPrevious != flags.first || clip->glideConnectedToNext != flags.second;
        if (apply) { clip->glideConnectedFromPrevious = flags.first; clip->glideConnectedToNext = flags.second; }
    }
    return changed;
}
}

void ProjectModel::toggleNoteConnection(const juce::String& noteId)
{
    bool changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks)
        {
            const auto ordered = orderedConnectionEndpoints(track);
            for (std::size_t i = 1; i < ordered.size(); ++i)
                if (ordered[i].note->id == noteId)
                {
                    auto connections = project.nativeConnections;
                    const bool migrated = materializeLegacyConnectionPairs(connections, ordered);
                    const bool connected = hasConnectionPair(connections, ordered[i - 1], ordered[i]);
                    if (!connected && !validNewConnectionPair(ordered[i - 1], ordered[i])) return;
                    const bool edited = mutateNativeConnectionPair(connections, ordered[i - 1], ordered[i], !connected);
                    if (!migrated && !edited && !projectConnectionFlags(connections, ordered, false)) return;
                    pushUndoLocked();
                    project.nativeConnections = std::move(connections);
                    projectConnectionFlags(project.nativeConnections, ordered, true);
                    changed = true;
                    break;
                }
            if (changed) break;
        }
    }
    if (changed) sendChangeMessage();
}


bool ProjectModel::setNativeAudioOverlap(const juce::String& trackId,bool enabled)
{
    {
        const juce::ScopedLock guard(lock);
        const auto found=std::find_if(project.tracks.begin(),project.tracks.end(),[&](const auto& t){return t.id==trackId;});
        if(found==project.tracks.end()||!trackShowsAllNativeRegions(*found))return false;
        auto next=*found;const auto cropped=!enabled&&removeNativeAudioOverlaps(next);
        if(!cropped&&next.allowNativeAudioOverlap==enabled)return false;
        // Cropping an incoming head also removes its old synthetic join.
        // The outgoing endpoint must not retain a compatibility glide flag.
        std::vector<juce::String> cutHeads;
        for(const auto& oldClip:found->clips)for(const auto& oldNote:oldClip.notes)
        {
            bool retained=false;
            for(const auto& newClip:next.clips)for(const auto& newNote:newClip.notes)if(newNote.id==oldNote.id)
            {retained=true;if(newNote.durationSeconds<oldNote.durationSeconds-1.e-9)cutHeads.push_back(oldNote.id);}
            if(!retained)cutHeads.push_back(oldNote.id);
        }
        next.allowNativeAudioOverlap=enabled;pushUndoLocked();*found=std::move(next);
        const auto cut=[&](const auto& id){return std::find(cutHeads.begin(),cutHeads.end(),id)!=cutHeads.end();};
        for(const auto& link:project.nativeConnections)if(cut(link.rightNoteId)||cut(link.leftNoteId))
            for(auto& t:project.tracks)for(auto& c:t.clips)for(auto& n:c.notes)
            {
                if(n.id==link.leftNoteId){n.connectedToNext=false;c.glideConnectedToNext=false;}
                if(n.id==link.rightNoteId){n.connectedToPrevious=false;c.glideConnectedFromPrevious=false;}
            }
        std::erase_if(project.nativeConnections,[&](const auto& c){return cut(c.leftNoteId)||cut(c.rightNoteId);});
    }
    sendChangeMessage();return true;
}

bool ProjectModel::linkNativeAudio(const std::vector<juce::String>& ids)
{
    {
        const juce::ScopedLock guard(lock);
        if(!nativeAudioLinkAvailable(project,ids))return false;
        auto updated=project;
        const auto selected=[&](const auto& n){return std::find(ids.begin(),ids.end(),n.id)!=ids.end();};
        for(auto& track:updated.tracks)
        {
            std::vector<ClipData> kept,chosen;bool affected=false;
            for(const auto& clip:track.clips)
            {
                if(std::none_of(clip.notes.begin(),clip.notes.end(),selected)){kept.push_back(clip);continue;}
                affected=true;auto parts=disconnectedNativeClip(clip,ids);if(!parts)return false;
                for(auto part:*parts)
                {
                    part.id=makeId("clip");
                    if(std::any_of(part.notes.begin(),part.notes.end(),selected))chosen.push_back(std::move(part));
                    else kept.push_back(std::move(part));
                }
            }
            if(!affected)continue;
            if(chosen.size()<2)return false;
            auto linked=assembledLinkedAudio(chosen);linked.id=makeId("clip");kept.push_back(std::move(linked));
            track.clips=std::move(kept);
        }
        pushUndoLocked();project=std::move(updated);
    }
    sendChangeMessage();return true;
}

bool ProjectModel::disconnectNativeAudio(const std::vector<juce::String>& ids)
{
    {
        const juce::ScopedLock guard(lock);
        if (!nativeAudioDisconnectAvailable(project,ids)) return false;
        // Prepare the whole edit before touching live data: one undo step, and
        // an invalid source/cut cannot leave half a selection disconnected.
        auto updated=project;
        const auto selected=[&](const auto& id){return std::find(ids.begin(),ids.end(),id)!=ids.end();};
        for (auto& track : updated.tracks)
        {
            if (!trackShowsAllNativeRegions(track)) continue;
            const auto envelopes = nativeSharedEnvelopes(track);
            for (auto& clip : track.clips)
                if (std::any_of(clip.notes.begin(), clip.notes.end(), [&](const auto& n) { return selected(n.id); }))
                    for (auto& note : clip.notes) materializeNativeSharedEnvelope(note, envelopes);
            struct Placed { NoteData* note; ClipData* clip; double start; };
            std::vector<Placed> ordered;
            for (auto& clip : track.clips) for (auto& note : clip.notes)
                ordered.push_back({&note,&clip,clip.startSeconds+note.startSeconds});
            std::stable_sort(ordered.begin(),ordered.end(),[](const auto& a,const auto& b){return a.start<b.start;});
            for (std::size_t i=1;i<ordered.size();++i)
                if (selected(ordered[i-1].note->id) || selected(ordered[i].note->id))
                {
                    ordered[i-1].note->connectedToNext=false;
                    ordered[i].note->connectedToPrevious=false;
                    ordered[i-1].clip->glideConnectedToNext=false;
                    ordered[i].clip->glideConnectedFromPrevious=false;
                }
            for (auto& item : ordered) if (selected(item.note->id))
            {
                item.note->connectedToPrevious=item.note->connectedToNext=false;
                item.note->utauAutoPitchTransition=false;
                item.clip->glideConnectedFromPrevious=item.clip->glideConnectedToNext=false;
            }
        }
        for (auto& track : updated.tracks) for (auto& clip : track.clips) for (auto& note : clip.notes)
            for (const auto& link : updated.nativeConnections)
                if (selected(link.leftNoteId) || selected(link.rightNoteId))
                {
                    if (note.id==link.leftNoteId) {note.connectedToNext=false;clip.glideConnectedToNext=false;}
                    if (note.id==link.rightNoteId) {note.connectedToPrevious=false;clip.glideConnectedFromPrevious=false;}
                }
        std::erase_if(updated.nativeConnections,[&](const auto& link)
            {return selected(link.leftNoteId)||selected(link.rightNoteId);});
        for (auto& track : updated.tracks)
        {
            if (!trackShowsAllNativeRegions(track)) continue;
            std::vector<ClipData> clips;
            for (const auto& clip : track.clips)
            {
                if (std::none_of(clip.notes.begin(),clip.notes.end(),[&](const auto& n){return selected(n.id);}))
                {clips.push_back(clip);continue;}
                auto parts=disconnectedNativeClip(clip,ids);if (!parts) return false;
                for (std::size_t i=0;i<parts->size();++i)
                {
                    auto part=std::move((*parts)[i]);part.id=i==0?clip.id:makeId("clip");
                    for (auto& note : part.notes) note.clipPartId.clear();
                    clips.push_back(std::move(part));
                }
            }
            track.clips=std::move(clips);
        }
        pushUndoLocked();project=std::move(updated);
    }
    sendChangeMessage();return true;
}

void ProjectModel::setNotesConnection(const std::vector<juce::String>& noteIds,
                                       bool enabled)
{
    if (noteIds.size() < 2) return;
    const auto includes = [&](const juce::String& id)
    { return std::find(noteIds.begin(), noteIds.end(), id) != noteIds.end(); };
    bool changed = false;
    {
        const juce::ScopedLock guard(lock);
        struct Group
        {
            std::vector<ConnectionEndpoint> ordered, selected;
        };
        std::vector<Group> groups;
        std::vector<juce::String> foundIds;
        for (auto& track : project.tracks)
        {
            Group group; group.ordered = orderedConnectionEndpoints(track);
            for (const auto& endpoint : group.ordered)
                if (includes(endpoint.note->id))
                {
                    group.selected.push_back(endpoint);
                    foundIds.push_back(endpoint.note->id);
                }
            if (group.selected.size() >= 2) groups.push_back(std::move(group));
        }
        // Invalid requests are atomic no-ops, not partial edits of the IDs that
        // happened to resolve. A pair spanning tracks never enters a group.
        if (groups.empty() || std::any_of(noteIds.begin(), noteIds.end(), [&](const auto& id)
            { return id.isEmpty() || std::find(foundIds.begin(), foundIds.end(), id) == foundIds.end(); })) return;
        auto connections = project.nativeConnections;
        for (const auto& group : groups)
        {
            changed |= materializeLegacyConnectionPairs(connections, group.ordered);
            for (std::size_t i = 1; i < group.selected.size(); ++i)
            {
                const auto& left = group.selected[i - 1]; const auto& right = group.selected[i];
                if (enabled && !hasConnectionPair(connections, left, right) && !validNewConnectionPair(left, right)) return;
                changed |= mutateNativeConnectionPair(connections, left, right, enabled);
            }
            changed |= projectConnectionFlags(connections, group.ordered, false);
        }
        if (!changed) return;
        pushUndoLocked();
        project.nativeConnections = std::move(connections);
        for (const auto& group : groups) projectConnectionFlags(project.nativeConnections, group.ordered, true);
    }
    if (changed) sendChangeMessage();
}

bool ProjectModel::setNotesPortamento(const std::vector<juce::String>& ids,bool enabled,juce::String& error,double durationSeconds)
{
    error.clear();if(ids.size()<2){error="Select at least two adjacent notes";return false;}
    std::set<juce::String> selected(ids.begin(),ids.end());
    if(selected.size()!=ids.size()||selected.count(juce::String{})){error="Duplicate or empty note identity";return false;}
    if(enabled&&(!std::isfinite(durationSeconds)||durationSeconds<.002||durationSeconds>30))
    {error="Portamento duration must be between 2ms and 30 seconds";return false;}
    bool changed=false;
    {
        const juce::ScopedLock guard(lock);TrackData* owner=nullptr;std::size_t count=0;
        for(auto& t:project.tracks)for(const auto& p:t.clips)for(const auto& n:p.notes)if(selected.count(n.id))
        {
            if(owner&&owner!=&t){error="Portamento endpoints must be on the same track";return false;}
            owner=&t;++count;
        }
        if(!owner||count!=ids.size()){error="Missing or ambiguous note identity";return false;}
        auto graph=project.nativeConnections;
        if(!enabled)
        {
            for(auto it=graph.begin();it!=graph.end();)
            {
                if(!selected.count(it->leftNoteId)||!selected.count(it->rightNoteId)||it->pitchCurve.empty()){++it;continue;}
                changed=true;
                if(it->type=="pitch-portamento")it=graph.erase(it);
                else{it->pitchCurve.clear();it->pitchCurveEndpointRelative=false;++it;} // retain prior physical/source intent
            }
        }
        else
        {
            const auto endpoints=graphPitchEndpoints(*owner);std::vector<std::size_t> order;
            for(std::size_t i=0;i<endpoints.size();++i)if(selected.count(endpoints[i].note->id))order.push_back(i);
            if(order.size()!=ids.size()){error="Ambiguous pitch endpoint";return false;}
            // Existing legacy source joins remain exactly as they were. Capture
            // their graph identity, but never project new pitch-only edges back
            // into concatenation, note splice or amplitude compatibility flags.
            changed|=materializeLegacyConnectionPairs(graph,orderedConnectionEndpoints(*owner));
            std::vector<juce::String> touched;
            for(std::size_t i=1;i<order.size();++i)
            {
                const auto a=order[i-1],b=order[i];error=graphPitchPairIssue(*owner,endpoints,a,b);if(error.isNotEmpty())return false;
                const auto& left=endpoints[a];const auto& right=endpoints[b];
                if(std::any_of(graph.begin(),graph.end(),[&](const auto& c)
                    {return (c.leftNoteId==left.note->id&&c.rightNoteId!=right.note->id)||(c.rightNoteId==right.note->id&&c.leftNoteId!=left.note->id);}))
                {error="A directional endpoint already has another connection; disconnect it explicitly first";return false;}
                auto edge=std::find_if(graph.begin(),graph.end(),[&](const auto& c){return c.leftNoteId==left.note->id&&c.rightNoteId==right.note->id;});
                if(edge==graph.end())
                {
                    ConnectionEndpoint l{const_cast<NoteData*>(left.note),const_cast<ClipData*>(left.parent),left.start};
                    ConnectionEndpoint r{const_cast<NoteData*>(right.note),const_cast<ClipData*>(right.parent),right.start};
                    mutateNativeConnectionPair(graph,l,r,true);edge=std::prev(graph.end());edge->type="pitch-portamento";edge->boundarySeconds=0;changed=true;
                }
                if(edge->pitchCurve.empty())
                {
                    edge->pitchCurveEndpointRelative=true;
                    edge->pitchStartOffsetSeconds=-std::min(durationSeconds*.5,(left.end-left.start)*.25);
                    edge->pitchEndOffsetSeconds=std::min(durationSeconds*.5,(right.end-right.start)*.25);
                    edge->pitchCurve={{0,0,PitchCurveShape::linear},{1,0,PitchCurveShape::smooth}};changed=true;
                }
                touched.push_back(edge->id);
            }
            const auto resolved=resolvedConnectionPitchCurves(*owner,graph);
            for(const auto& id:touched)
            {
                const auto c=std::find_if(resolved.begin(),resolved.end(),[&](const auto& c){return c.connectionId==id;});
                if(c==resolved.end()||!c->active){error=c==resolved.end()?juce::String("Unresolvable portamento"):c->issue;return false;}
            }
        }
        if(!changed)return false;
        pushUndoLocked();project.nativeConnections=std::move(graph);
    }
    sendChangeMessage();return true;
}

bool ProjectModel::setConnectionPitchCurve(const juce::String& id,std::vector<PitchCurveEditPoint> points,juce::String& error)
{
    error.clear();bool changed=false;
    {
        const juce::ScopedLock guard(lock);auto graph=project.nativeConnections;
        const auto* track=graphConnectionTrack(project,graph,id,error);if(!track)return false;
        changed=prepareConnectionPitchEdit(*track,graph,id,points,error);
        if(changed){pushUndoLocked();project.nativeConnections=std::move(graph);}
    }
    if(changed)sendChangeMessage();return changed;
}

bool ProjectModel::setConnectionPitchPoint(const juce::String& id,int index,PitchCurveEditPoint point,juce::String& error)
{
    error.clear();const juce::ScopedLock guard(lock);
    const auto* track=graphConnectionTrack(project,project.nativeConnections,id,error);if(!track)return false;
    const auto curves=resolvedConnectionPitchCurves(*track,project.nativeConnections);
    const auto c=std::find_if(curves.begin(),curves.end(),[&](const auto& c){return c.connectionId==id;});
    if(c==curves.end()||!c->active||index<0||index>=static_cast<int>(c->points.size()))
    {error="The editable connection point is unavailable";return false;}
    auto points=c->points;points[static_cast<std::size_t>(index)]=point;
    return setConnectionPitchCurve(id,std::move(points),error);
}

namespace
{
struct LocalMaterialView
{
    const TrackData* track=nullptr;const ClipData* parent=nullptr;const ClipData* owner=nullptr;const NoteData* note=nullptr;
    double noteOffset=0;
};
bool localMaterialView(const ProjectData& project,const juce::String& id,LocalMaterialView& view,juce::String& error)
{
    int matches=0;
    for(const auto& track:project.tracks)for(const auto& clip:track.clips)for(const auto& note:clip.notes)if(note.id==id)
    {++matches;view.track=&track;view.parent=&clip;view.note=&note;}
    if(matches!=1){error=matches?"Ambiguous note identity":"Note no longer exists";return false;}
    if(view.parent->parts.empty())view.owner=view.parent;
    else for(const auto& part:view.parent->parts)if(part.id==view.note->clipPartId)
    {if(view.owner){error="Ambiguous source-child ownership";return false;}view.owner=&part;view.noteOffset=part.startSeconds;}
    if(!view.owner){error="Missing explicit source-child ownership";return false;}return true;
}
SampleRegionSetting localMaterialRow(const NoteData& note,const ClipData& owner,double offset)
{
    auto local=note;local.startSeconds-=offset;
    auto row=nativeNoteSourceTiming(local,owner,nativeClipClock(owner));
    row.name=note.materialBoundAlias.isNotEmpty()?note.materialBoundAlias:note.label;
    row.role=note.nativeRole;row.nativeUnpitched=note.nativeUnpitched;row.provenance=note.nativeProvenance;row.confidence=note.nativeConfidence;
    row.materialAudioId=note.materialAudioId;row.materialRegionId=note.materialRegionId;row.additionalAliases=note.materialAdditionalAliases;
    row.connectedNextRegionId=note.materialConnectedNextRegionId;
    row.overlapSeconds=note.utauOverlapOverrideEnabled||std::abs(note.utauOverlapSeconds)>1.e-12?note.utauOverlapSeconds
        :note.nativeSegments.empty()?0:note.nativeSegments.front().overlapSeconds;
    row.melodynePitchCenterCents=note.midiNote*100;row.melodyneOriginalPitchCenterCents=note.sourceMidiCenter>=0?note.sourceMidiCenter*100:0;
    row.relativePitchCents=note.sourceMidiCenter>=0?(note.midiNote-note.sourceMidiCenter)*100:0;
    row.melodynePitchDrift=note.drift;row.melodynePitchModulation=note.modulation;row.melodyneFormantCents=note.formantSemitones*100;
    row.melodyneAmplitude=note.gain;row.melodyneSibilantBalance=note.breath;row.amplitudeEnvelope=note.amplitudeEnvelope;
    return row;
}
bool localMaterialMusicUnchanged(const SampleRegionSetting& a,const SampleRegionSetting& b)
{
    if(a.relativePitchCents!=b.relativePitchCents||a.melodyneData!=b.melodyneData||a.melodynePitchCenterCents!=b.melodynePitchCenterCents
        ||a.melodyneOriginalPitchCenterCents!=b.melodyneOriginalPitchCenterCents||a.melodynePitchDrift!=b.melodynePitchDrift
        ||a.melodynePitchModulation!=b.melodynePitchModulation||a.melodyneTransitionSeconds!=b.melodyneTransitionSeconds
        ||a.melodyneFormantCents!=b.melodyneFormantCents||a.melodyneAmplitude!=b.melodyneAmplitude
        ||a.melodyneSibilantBalance!=b.melodyneSibilantBalance||a.melodyneAttackSeconds!=b.melodyneAttackSeconds
        ||a.melodyneDecayElongation!=b.melodyneDecayElongation||a.amplitudeEnvelope.size()!=b.amplitudeEnvelope.size())return false;
    for(std::size_t i=0;i<a.amplitudeEnvelope.size();++i)
    {const auto& x=a.amplitudeEnvelope[i];const auto& y=b.amplitudeEnvelope[i];if(x.timeSeconds!=y.timeSeconds||x.gainDb!=y.gainDb||x.linearToNext!=y.linearToNext||x.nativeSeamAnchor!=y.nativeSeamAnchor)return false;}
    return true;
}
bool localMaterialIntact(const NoteData& note,const SampleRegionSetting& row)
{
    const auto start=note.materialSourceStartSeconds>=0?note.materialSourceStartSeconds:note.nativeSourceStartSeconds;
    const auto end=note.materialSourceEndSeconds>=0?note.materialSourceEndSeconds:note.nativeSourceEndSeconds;
    return (start<0||std::abs(start-row.regionStartSeconds)<=1.e-8)&&(end<0||std::abs(end-row.regionEndSeconds)<=1.e-8);
}
}

bool ProjectModel::getNoteMaterialLocalBinding(const juce::String& noteId,
    NoteMaterialLocalBinding& binding,SampleRegionSetting& row,juce::String& error) const
{
    error.clear();binding=NoteMaterialLocalBinding{};row=SampleRegionSetting{};
    {
        const juce::ScopedLock guard(lock);LocalMaterialView view;
        if(!localMaterialView(project,noteId,view,error))return false;
        binding.noteId=noteId;binding.parentClipId=view.parent->id;binding.sourcePartId=view.parent->parts.empty()?juce::String{}:view.owner->id;
        binding.audioId=view.note->materialAudioId;binding.regionId=view.note->materialRegionId;binding.projectRevision=revision;
        row=localMaterialRow(*view.note,*view.owner,view.noteOffset);
        if(trackUsesVoicebankSynthesis(*view.track))
        {
            binding.utauScalarTiming=!trackIsDiffSinger(*view.track)&&view.track->utauMode==UtauMode::classic;
            binding.diagnostic=binding.utauScalarTiming?"Classic scalar OTO timing is supported, but this note needs an authoritative catalog audio/region binding; no alias guess was made."
                :"This renderer has no full native row annotation adapter.";
            return true;
        }
        binding.sourceFile=view.owner->sourceFile;
        binding.bindingResolved=trackShowsAllNativeRegions(*view.track)&&binding.sourceFile!=juce::File{}&&binding.audioId.isNotEmpty()&&binding.regionId.isNotEmpty();
        binding.nativeFullRow=binding.bindingResolved&&localMaterialIntact(*view.note,row);
        if(!binding.bindingResolved)binding.diagnostic="A stable native audio/region binding is required; legacy rows are not guessed by alias or order.";
        else if(!binding.nativeFullRow)binding.diagnostic="The note is a cropped source slice; full-row timing cannot expand or replace its protected source ownership.";
    }
    if(!binding.bindingResolved)return true;
    if(!binding.sourceFile.existsAsFile())
    {binding.bindingResolved=binding.nativeFullRow=false;binding.diagnostic="The bound source audio is missing.";return true;}
    const auto sourceSize=binding.sourceFile.getSize();const auto sourceModified=binding.sourceFile.getLastModificationTime();
    binding.sourceFingerprint=juce::SHA256(binding.sourceFile).toHexString();
    if(binding.sourceFile.getSize()!=sourceSize||binding.sourceFile.getLastModificationTime()!=sourceModified)
    {error="Source audio changed while opening its local annotation.";return false;}
    {
        const juce::ScopedLock guard(lock);LocalMaterialView view;
        if(revision!=binding.projectRevision||!localMaterialView(project,noteId,view,error)
            ||view.parent->id!=binding.parentClipId||view.owner->sourceFile!=binding.sourceFile
            ||view.note->materialAudioId!=binding.audioId||view.note->materialRegionId!=binding.regionId)
        {error="The note binding changed while opening its local annotation.";return false;}
        if(view.owner->materialSourceFingerprint.isNotEmpty()&&view.owner->materialSourceFingerprint!=binding.sourceFingerprint)
        {error="The bound source audio changed; refresh its acoustic facts before local editing.";return false;}
    }
    return true;
}

bool ProjectModel::makeClassicNoteMaterialOto(const SampleRegionSetting& input,backend::UtauOtoOverride& result,juce::String& error)
{
    error.clear();
    if(input.pronunciationKind!=PronunciationKind::normal||input.simplifiedConsonant||input.vowelSplitsExplicit)
    {error="Classic OTO supports scalar P/C/overlap timing only, not native kind/link intent or named/weighted vowel subdivisions.";return false;}
    if(!std::isfinite(input.regionStartSeconds)||!std::isfinite(input.regionEndSeconds)||!std::isfinite(input.alignmentSeconds)
        ||!std::isfinite(input.fixedDurationSeconds)||!std::isfinite(input.overlapSeconds)||input.regionStartSeconds<0
        ||input.regionEndSeconds<=input.regionStartSeconds||input.alignmentSeconds<input.regionStartSeconds||input.alignmentSeconds>input.regionEndSeconds
        ||input.fixedDurationSeconds<0||input.fixedDurationSeconds>input.regionEndSeconds-input.regionStartSeconds)
    {error="Invalid scalar source timing.";return false;}
    VoicebankOtoEntry entry;entry.offsetMs=input.regionStartSeconds*1000;entry.consonantMs=input.fixedDurationSeconds*1000;
    entry.cutoffMs=-(input.regionEndSeconds-input.regionStartSeconds)*1000;entry.preutteranceMs=(input.alignmentSeconds-input.regionStartSeconds)*1000;
    entry.overlapMs=input.overlapSeconds*1000;result=SampleSettings::noteOtoFromEntry(entry,false);return true;
}

bool ProjectModel::applyNoteMaterialLocalRow(const NoteMaterialLocalBinding& binding,const SampleRegionSetting& row,
    std::uint64_t& resultingRevision,juce::String& error)
{
    return applyNoteMaterialLocalRowInternal(binding,row,false,resultingRevision,error);
}

bool ProjectModel::restoreNoteMaterialLocalRow(const NoteMaterialLocalBinding& binding,SampleRegionSetting& restored,
    std::uint64_t& resultingRevision,juce::String& error)
{
    error.clear();resultingRevision=0;
    if(!binding.bindingResolved||!binding.nativeFullRow){error="A resolved intact native binding is required to restore a full row.";return false;}
    const auto manifest=SampleSettings::annotationFor(binding.sourceFile);
    if(!manifest.existsAsFile()){error="The latest native source annotation is missing; local data was retained.";return false;}
    std::vector<SampleRegionSetting> latest;
    if(!OttoMaterialIO::loadAudio(manifest,binding.sourceFile,latest,error,nullptr,binding.audioId))return false;
    const SampleRegionSetting* match=nullptr;
    for(const auto& row:latest)if(row.materialAudioId==binding.audioId&&row.materialRegionId==binding.regionId)
    {if(match){error="Ambiguous source audio/region identity; local data was retained.";return false;}match=&row;}
    if(!match){error="The source pronunciation was deleted; local data was retained.";return false;}
    // Return the row from the SAME committed candidate. A second fallible
    // getter after publication could report false despite a completed undo.
    return applyNoteMaterialLocalRowInternal(binding,*match,true,resultingRevision,error,&restored);
}

bool ProjectModel::applyNoteMaterialLocalRowInternal(const NoteMaterialLocalBinding& binding,const SampleRegionSetting& input,
    bool restoringSource,std::uint64_t& resultingRevision,juce::String& error,SampleRegionSetting* committedRow)
{
    error.clear();resultingRevision=0;
    const auto reject=[&](const juce::String& problem){error=problem;return false;};
    if(!binding.bindingResolved||!binding.nativeFullRow||binding.audioId.isEmpty()||binding.regionId.isEmpty()||binding.sourceFingerprint.isEmpty())
        return reject("Full-row local editing requires an intact, authoritative native source binding. Classic OTO remains a separate scalar capability.");
    if(input.materialAudioId!=binding.audioId||input.materialRegionId!=binding.regionId)return reject("Local edits cannot replace source identity.");
    if(!binding.sourceFile.existsAsFile())return reject("The bound source audio is missing.");
    const auto sourceBytes=binding.sourceFile.getSize();const auto sourceTime=binding.sourceFile.getLastModificationTime();
    if(juce::SHA256(binding.sourceFile).toHexString()!=binding.sourceFingerprint)
        return reject("Source audio changed while the local annotation was open.");
    double mediaDuration=0;
    {
        juce::AudioFormatManager formats;formats.registerBasicFormats();std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(binding.sourceFile));
        if(!reader||reader->sampleRate<=0)return reject("The bound source is not readable audio.");
        mediaDuration=reader->lengthInSamples/reader->sampleRate;
    }
    auto row=input;
    if(!std::isfinite(row.regionStartSeconds)||!std::isfinite(row.regionEndSeconds)||!std::isfinite(row.alignmentSeconds)
        ||!std::isfinite(row.fixedDurationSeconds)||!std::isfinite(row.overlapSeconds)||!std::isfinite(row.confidence)
        ||row.regionStartSeconds<0||row.regionEndSeconds<=row.regionStartSeconds||row.regionEndSeconds>mediaDuration+1.e-8
        ||row.alignmentSeconds<row.regionStartSeconds||row.alignmentSeconds>row.regionEndSeconds
        ||row.fixedDurationSeconds<0||row.fixedDurationSeconds>row.regionEndSeconds-row.regionStartSeconds
        ||(row.simplifiedConsonant&&row.regionStartSeconds+row.fixedDurationSeconds<row.alignmentSeconds-1.e-8))
        return reject("Invalid source pronunciation timing.");
    if(row.vowelSplitsExplicit)
    {
        std::set<juce::String> ids;
        for(const auto& segment:row.segments)
        {
            if(segment.id.isNotEmpty()&&!ids.insert(segment.id).second)return reject("Duplicate manual segment identity.");
            if(segment.role==NativeSegmentRole::vowel&&(segment.alias.trim().isEmpty()||segment.alias.trim()=="-"))return reject("Manual vowel subdivisions must be named.");
        }
    }
    if(!applyPronunciationTiming(row,pronunciationTimingFor(row),error))return false;
    {
        const juce::ScopedLock guard(lock);LocalMaterialView view;
        if(revision!=binding.projectRevision)return reject("The project changed while the local annotation was open.");
        if(!localMaterialView(project,binding.noteId,view,error))return false;
        if(view.parent->id!=binding.parentClipId||(view.parent->parts.empty()?juce::String{}:view.owner->id)!=binding.sourcePartId
            ||view.owner->sourceFile!=binding.sourceFile||view.note->materialAudioId!=binding.audioId||view.note->materialRegionId!=binding.regionId
            ||!trackShowsAllNativeRegions(*view.track)||trackUsesVoicebankSynthesis(*view.track))return reject("The physical note binding changed.");
        if(binding.sourceFile.getSize()!=sourceBytes||binding.sourceFile.getLastModificationTime()!=sourceTime)return reject("Source audio changed during validation.");
        if(view.owner->materialSourceFingerprint.isNotEmpty()&&view.owner->materialSourceFingerprint!=binding.sourceFingerprint)
            return reject("Source acoustic facts refer to a different recording fingerprint.");
        const auto current=localMaterialRow(*view.note,*view.owner,view.noteOffset);
        if(!localMaterialIntact(*view.note,current))return reject("A cropped pronunciation cannot be expanded or rebound by local full-row editing.");
        if(!restoringSource&&!localMaterialMusicUnchanged(input,current))return reject("Use the musical editors to change pitch, gain, formant or automation; local annotation cannot reset them.");
        if(!restoringSource&&!view.note->materialAliasFollowsPrimary&&row.name!=current.name)return reject("The selected secondary alias is pinned; local annotation cannot silently replace it.");
        auto candidate=*view.parent;NoteData* selected=nullptr;
        for(auto& note:candidate.notes)if(note.id==binding.noteId)selected=&note;
        auto* owner=&candidate;
        if(!candidate.parts.empty())
        {owner=nullptr;for(auto& part:candidate.parts)if(part.id==binding.sourcePartId)owner=&part;}
        if(!selected||!owner)return reject("Source ownership changed.");
        const auto oldClock=nativeClipClock(*owner);const auto oldOffset=owner->sourceOffsetSeconds;
        if(oldClock.size()<2)return reject("Missing source clock.");
        for(std::size_t i=0;i<oldClock.size();++i)
            if(!std::isfinite(oldClock[i].targetSeconds)||!std::isfinite(oldClock[i].sourceSeconds)
                ||(i&&(oldClock[i].targetSeconds<=oldClock[i-1].targetSeconds||oldClock[i].sourceSeconds<=oldClock[i-1].sourceSeconds)))
                return reject("A non-monotonic source clock cannot be locally retimed.");
        const auto start=selected->startSeconds-view.noteOffset,end=start+selected->durationSeconds;
        if(start<oldClock.front().targetSeconds-1.e-8||end>oldClock.back().targetSeconds+1.e-8)return reject("The note extends outside its audible source owner.");
        const auto zones=pronunciationSegmentsFor(row),oldZones=pronunciationSegmentsFor(current);
        bool retime=row.regionStartSeconds!=current.regionStartSeconds||row.regionEndSeconds!=current.regionEndSeconds
            ||row.alignmentSeconds!=current.alignmentSeconds||row.fixedDurationSeconds!=current.fixedDurationSeconds||zones.size()!=oldZones.size();
        if(!retime)for(std::size_t i=0;i<zones.size();++i)
            retime|=zones[i].sourceStartSeconds!=oldZones[i].sourceStartSeconds||zones[i].sourceEndSeconds!=oldZones[i].sourceEndSeconds
                ||zones[i].role!=oldZones[i].role||zones[i].stretchable!=oldZones[i].stretchable||zones[i].stretchWeight!=oldZones[i].stretchWeight;
        if(retime)
        {
            for(const auto& zone:zones)if(!zone.stretchable)for(const auto& oldZone:oldZones)if(!oldZone.stretchable)
            {
                // Protect the source that was AND remains fixed. A deliberate C
                // edit may newly protect part of the old vowel; its old elastic
                // ratio must not be mistaken for an already-stretched consonant.
                const auto a=std::max(row.regionStartSeconds+zone.sourceStartSeconds,current.regionStartSeconds+oldZone.sourceStartSeconds);
                const auto b=std::min(row.regionStartSeconds+zone.sourceEndSeconds,current.regionStartSeconds+oldZone.sourceEndSeconds);
                if(b>a+1.e-9)
                {
                    const auto heard=nativeTargetTimeAt(oldClock,b-oldOffset)-nativeTargetTimeAt(oldClock,a-oldOffset);
                    if(std::abs(heard-(b-a))>1.e-6)return reject("Existing protected source timing is non-unit; local retiming would stretch a fixed consonant/source zone.");
                }
            }
            std::vector<SourceTimePoint> localMap;
            if(!makePronunciationStretchMap(row,selected->durationSeconds,localMap,error))return false;
            std::vector<SourceTimePoint> absolute;
            for(auto point:oldClock)if(point.targetSeconds<start-1.e-9||point.targetSeconds>end+1.e-9)
            {point.sourceSeconds+=oldOffset;absolute.push_back(point);}
            for(auto point:localMap){point.targetSeconds+=start;absolute.push_back(point);}
            for(const auto& other:candidate.notes)
            {
                if(other.id==selected->id||(!candidate.parts.empty()&&other.clipPartId!=binding.sourcePartId))continue;
                const auto a=other.startSeconds-view.noteOffset,b=a+other.durationSeconds;
                if(std::min(end,b)>std::max(start,a)+1.e-8)return reject("Overlapping notes cannot independently own one shared source clock.");
                for(const auto t:{a,b})if(t>=oldClock.front().targetSeconds&&t<=oldClock.back().targetSeconds)
                    absolute.push_back({t,oldOffset+nativeSourceTimeAt(oldClock,t)});
            }
            std::stable_sort(absolute.begin(),absolute.end(),[](const auto& a,const auto& b){return a.targetSeconds<b.targetSeconds;});
            std::vector<SourceTimePoint> unique;
            for(const auto& point:absolute)
            {
                if(!unique.empty()&&std::abs(point.targetSeconds-unique.back().targetSeconds)<1.e-9)
                {if(std::abs(point.sourceSeconds-unique.back().sourceSeconds)>1.e-8)return reject("Local boundaries conflict with a pinned neighbouring note.");}
                else unique.push_back(point);
            }
            for(std::size_t i=1;i<unique.size();++i)if(unique[i].sourceSeconds<=unique[i-1].sourceSeconds+1.e-9)return reject("Local source boundaries overlap or reverse.");
            if(unique.size()<2||unique.front().sourceSeconds<0||unique.back().sourceSeconds>mediaDuration+1.e-8)return reject("Local timing leaves the source recording.");
            if(selected->sourcePitchMeasured&&(!owner->nativeSourcePitch||owner->nativeSourcePitch->empty()))
            {
                std::vector<NoteData> measured;
                for(auto note:candidate.notes)if(note.sourcePitchMeasured&&(candidate.parts.empty()||note.clipPartId==binding.sourcePartId))
                {note.startSeconds-=view.noteOffset;measured.push_back(std::move(note));}
                owner->nativeSourcePitch=nativeSourcePitchReference(measured,[&](double t){return oldOffset+nativeSourceTimeAt(oldClock,t);});
            }
            if(selected->sourcePitchMeasured&&(!nativeReferencePitchAt(*owner,row.regionStartSeconds)||!nativeReferencePitchAt(*owner,row.regionEndSeconds)))
                return reject("The changed interval lacks measured source pitch coverage; analyze the source before expanding it.");
            const auto first=unique.front().sourceSeconds,last=unique.back().sourceSeconds;
            for(auto& point:unique)point.sourceSeconds-=first;
            if(!publishMaterialSourceClock(*owner,oldClock,std::move(unique),first,last,error))return false;
            if(selected->sourcePitchMeasured)
            {
                auto local=*selected;local.startSeconds-=view.noteOffset;refreshNativeTrimPitch(local,*owner);
                selected->contour=std::move(local.contour);selected->sourceMidiCenter=local.sourceMidiCenter;selected->sourcePitchMeasured=local.sourcePitchMeasured;
            }
        }
        const auto newClock=nativeClipClock(*owner);const auto unresolved=selected->materialReferenceUnresolved;
        applyMaterialAnnotationSnapshot(*selected,row,true,SampleSettings::nativeSegmentsFor(row));
        selected->nativeUnpitched=row.nativeUnpitched;selected->materialLocalOverride=!restoringSource;
        if(!restoringSource)selected->materialReferenceUnresolved=unresolved;
        selected->consonantSeconds=nativeTargetTimeAt(newClock,row.regionStartSeconds+row.fixedDurationSeconds-owner->sourceOffsetSeconds)-start;
        selected->utauPreutteranceOverrideEnabled=true;selected->utauPreutteranceSeconds=row.alignmentSeconds-row.regionStartSeconds;
        selected->utauOverlapOverrideEnabled=true;selected->utauOverlapSeconds=row.overlapSeconds;
        owner->materialSourceFingerprint=binding.sourceFingerprint;
        if(committedRow)*committedRow=localMaterialRow(*selected,*owner,view.noteOffset);
        pushUndoLocked();*const_cast<ClipData*>(view.parent)=std::move(candidate);resultingRevision=revision;
    }
    sendChangeMessage();return true;
}

void ProjectModel::setNoteMaterialLocalOverride(const juce::String& noteId, bool enabled)
{
    bool changed = false;
    {
        const juce::ScopedLock guard(lock);
        for (auto& track : project.tracks) for (auto& clip : track.clips)
            for (auto& note : clip.notes)
                if (note.id == noteId && note.materialLocalOverride != enabled)
                {
                    if (!changed) pushUndoLocked();
                    note.materialLocalOverride = enabled;
                    changed = true;
                }
    }
    if (changed) sendChangeMessage();
}

void ProjectModel::applySourceSettings(const juce::File& source,
                                      const std::vector<SampleRegionSetting>& rows)
{
    (void) applySourceSettingsWithReport(source, rows);
}

// Exact cubic crop: unlike splitNote's historical linear cut, this retains
// shaped pitch segments. Each fragment owns only its local handles; existing
// native shared-line endpoint deduplication gives each seam ONE authority.
bool sliceNativePitchCurve(const std::vector<PitchCurveEditPoint>& points, double from, double to,
                        std::vector<PitchCurveEditPoint>& result)
{
    result.clear();
    if (points.empty() || to <= from) return false;
    struct P { double x, y; };
    struct Cubic { P a, b, c, d; };
    const auto lerp = [](P a, P b, double t) { return P{a.x+(b.x-a.x)*t,a.y+(b.y-a.y)*t}; };
    const auto split = [&](Cubic s, double t)
    {
        const auto ab=lerp(s.a,s.b,t),bc=lerp(s.b,s.c,t),cd=lerp(s.c,s.d,t);
        const auto abc=lerp(ab,bc,t),bcd=lerp(bc,cd,t),m=lerp(abc,bcd,t);
        return std::pair{Cubic{s.a,ab,abc,m},Cubic{m,bcd,cd,s.d}};
    };
    const auto parameter = [&](const Cubic& s, double x)
    {
        double lo=0,hi=1;
        for(int i=0;i<52;++i){const auto t=(lo+hi)*.5;if(split(s,t).first.d.x<x)lo=t;else hi=t;}
        return (lo+hi)*.5;
    };
    std::vector<double> intervals(points.size()>1?points.size()-1:0);
    std::vector<float> secants(intervals.size()), tangents(points.size(),0);
    for(std::size_t i=0;i<intervals.size();++i)
    {
        intervals[i]=points[i+1].timeSeconds-points[i].timeSeconds;
        if(intervals[i]<=1.e-9)return false;
        secants[i]=(points[i+1].targetMidi-points[i].targetMidi)/static_cast<float>(intervals[i]);
    }
    for(std::size_t i=1;i+1<points.size();++i)
        if(secants[i-1]!=0 && secants[i]!=0 && std::signbit(secants[i-1])==std::signbit(secants[i]))
        {
            const auto w1=2*intervals[i]+intervals[i-1],w2=intervals[i]+2*intervals[i-1];
            tangents[i]=static_cast<float>((w1+w2)/(w1/secants[i-1]+w2/secants[i]));
        }
    const auto emit = [&](Cubic s) -> bool
    {
        if(s.d.x<=from || s.a.x>=to)return true;
        if(s.d.x>to)s=split(s,parameter(s,to)).first;
        if(s.a.x<from)s=split(s,parameter(s,from)).second;
        const auto dx=s.d.x-s.a.x,dy=s.d.y-s.a.y;
        if(dx<=1.e-10)return false;
        if(result.empty())result.push_back({s.a.x-from,static_cast<float>(s.a.y),PitchCurveShape::linear});
        PitchCurveEditPoint end{s.d.x-from,static_cast<float>(s.d.y),PitchCurveShape::customBezier};
        if(std::abs(dy)<1.e-8)
        {
            // Equal-valued ends with an interior excursion cannot be encoded by
            // normalized endpoint-relative controls. Reject rather than flatten.
            if(std::abs(s.b.y-s.a.y)>1.e-7 || std::abs(s.c.y-s.a.y)>1.e-7)return false;
            end.shape=PitchCurveShape::linear;
        }
        else
        {
            end.bezierX1=static_cast<float>(std::clamp((s.b.x-s.a.x)/dx,0.0,1.0));
            end.bezierX2=static_cast<float>(std::clamp((s.c.x-s.a.x)/dx,0.0,1.0));
            end.bezierY1=static_cast<float>((s.b.y-s.a.y)/dy);
            end.bezierY2=static_cast<float>((s.c.y-s.a.y)/dy);
            if(!std::isfinite(end.bezierY1)||!std::isfinite(end.bezierY2))return false;
        }
        end.highPrecisionBezier=end.shape==PitchCurveShape::customBezier;
        result.push_back(end);return true;
    };
    const auto flat = [&](double a,double b,double y)
    {return emit({{a,y},{a+(b-a)/3,y},{a+2*(b-a)/3,y},{b,y}});};
    if(from<points.front().timeSeconds && !flat(from,std::min(to,points.front().timeSeconds),points.front().targetMidi))return false;
    for(std::size_t i=1;i<points.size();++i)
    {
        const auto& a=points[i-1];const auto& d=points[i];const auto dx=d.timeSeconds-a.timeSeconds;
        if(a.timeSeconds>=from&&d.timeSeconds<=to&&d.shape!=PitchCurveShape::natural)
        {
            // Uncut context-free segments retain their original handles and
            // legacy float evaluator exactly; only actual crops need new cubics.
            if(result.empty()){auto first=a;first.timeSeconds-=from;result.push_back(first);}
            auto end=d;end.timeSeconds-=from;result.push_back(end);continue;
        }
        const auto dy=static_cast<double>(d.targetMidi-a.targetMidi);
        double x1=1.0/3,x2=2.0/3,y1=1.0/3,y2=2.0/3;
        switch(d.shape)
        {
            case PitchCurveShape::smooth:y1=0;y2=1;break;
            case PitchCurveShape::easeIn:y1=0;y2=1.0/3;break;
            case PitchCurveShape::easeOut:y1=2.0/3;y2=1;break;
            case PitchCurveShape::customBezier:x1=std::clamp<double>(d.bezierX1,0,1);x2=std::clamp<double>(d.bezierX2,0,1);y1=d.bezierY1;y2=d.bezierY2;break;
            case PitchCurveShape::natural:
                if(std::abs(dy)>1.e-12){y1=tangents[i-1]*dx/(3*dy);y2=1-tangents[i]*dx/(3*dy);}break;
            default:break;
        }
        if(!emit({{a.timeSeconds,a.targetMidi},{a.timeSeconds+x1*dx,a.targetMidi+y1*dy},
            {a.timeSeconds+x2*dx,a.targetMidi+y2*dy},{d.timeSeconds,d.targetMidi}}))return false;
    }
    if(to>points.back().timeSeconds && !flat(std::max(from,points.back().timeSeconds),to,points.back().targetMidi))return false;
    return result.size()>=2;
}

namespace
{
// One curve-vector splice authority for both note-local and absolute graph
// editing. The wrappers only supply domains/ownership, never fake NoteData.
bool composePitchStrokeCurve(const std::vector<PitchCurveEditPoint>& original,
    const std::vector<PitchCurveEditPoint>& inputStroke,double minimum,double maximum,bool keepOutside,
    std::vector<PitchCurveEditPoint>& out,juce::String& error,bool preserveLeft,bool preserveRight)
{
    out.clear();error.clear();
    if(original.empty()||inputStroke.size()<2){error="A pitched stroke requires an original curve and at least two controls";return false;}
    auto stroke=inputStroke;
    std::stable_sort(stroke.begin(),stroke.end(),[](const auto& a,const auto& b){return a.timeSeconds<b.timeSeconds;});
    for(std::size_t i=0;i<stroke.size();++i)
    {
        if(!finiteGraphPoint(stroke[i])||stroke[i].targetMidi<0||stroke[i].targetMidi>127
            ||(i&&stroke[i].timeSeconds<=stroke[i-1].timeSeconds))
        {error="Invalid or duplicate stroke controls";return false;}
    }
    const auto from=stroke.front().timeSeconds,to=stroke.back().timeSeconds;
    if(!std::isfinite(minimum)||!std::isfinite(maximum)||to-from<1.e-6||from<minimum-1.e-10||to>maximum+1.e-10)
    {error="Stroke is outside the existing authored/body union domain";return false;}
    // A free edge is allowed only at an artificial union-domain clip. Never
    // destroy real old authored controls outside the overall user's gesture.
    if((!preserveLeft&&original.front().timeSeconds<from-1.e-10)
        ||(!preserveRight&&original.back().timeSeconds>to+1.e-10))
    {error="A free clipped edge would discard old outside authored support";return false;}
    // Stroke samples define a polyline. Guards must sample that NEW line at
    // their actual time, not shift endpoint Y horizontally by 1ms (which
    // changes every interior sample's slope).
    for(auto& p:stroke)p.shape=PitchCurveShape::linear;
    const auto guard=std::min(.001,(to-from)*.25);
    const auto innerLeft=from+(preserveLeft?guard:0),innerRight=to-(preserveRight?guard:0);
    std::vector<PitchCurveEditPoint> replacement;
    if(preserveLeft)replacement.push_back({from,evaluatePitchCurve(original,from),PitchCurveShape::linear});
    replacement.push_back({innerLeft,evaluatePitchCurve(stroke,innerLeft),PitchCurveShape::linear});
    for(auto p:stroke)if(p.timeSeconds>innerLeft&&p.timeSeconds<innerRight){p.shape=PitchCurveShape::linear;replacement.push_back(p);}
    replacement.push_back({innerRight,evaluatePitchCurve(stroke,innerRight),PitchCurveShape::linear});
    if(preserveRight)replacement.push_back({to,evaluatePitchCurve(original,to),PitchCurveShape::linear});
    if(!keepOutside){out=std::move(replacement);return true;}
    const auto append=[&](const std::vector<PitchCurveEditPoint>& points)
    {
        for(const auto& p:points)
        {
            // The already composed incoming descriptor owns a same-time seam.
            // Prefix incoming survives stroke-start, stroke incoming survives
            // suffix-start; replacing it with a first-point default linearizes.
            if(!out.empty()&&std::abs(out.back().timeSeconds-p.timeSeconds)<1.e-10)continue;
            out.push_back(p);
        }
    };
    if(preserveLeft&&original.front().timeSeconds<from)
    {
        std::vector<PitchCurveEditPoint> prefix;
        if(!sliceNativePitchCurve(original,original.front().timeSeconds,from,prefix)){error="Original cubic prefix cannot be cropped exactly";return false;}
        for(auto& p:prefix)p.timeSeconds+=original.front().timeSeconds;append(prefix);
    }
    append(replacement);
    if(preserveRight&&original.back().timeSeconds>to)
    {
        std::vector<PitchCurveEditPoint> suffix;
        if(!sliceNativePitchCurve(original,to,original.back().timeSeconds,suffix)){error="Original cubic suffix cannot be cropped exactly";out.clear();return false;}
        for(auto& p:suffix)p.timeSeconds+=to;append(suffix);
    }
    for(const auto& p:out)if(!finiteGraphPoint(p)||p.targetMidi<0||p.targetMidi>127)
    {error="Existing curve cannot remain exact inside the editable MIDI range";out.clear();return false;}
    return true;
}

} // curve-vector splice implementation

bool composeNativePitchStroke(const NoteData& note,const std::vector<PitchCurveEditPoint>& stroke,
    std::vector<PitchCurveEditPoint>& out,bool& storePitchControlPoints,juce::String& error,bool preserveLeft,bool preserveRight)
{
    out.clear();error.clear();storePitchControlPoints=!note.pitchControlPoints.empty();
    if(note.nativeUnpitched||!std::isfinite(note.durationSeconds)){error="Missing pitched note domain";return false;}
    const auto original=ownPitchPoints(note);if(original.empty()){error="Missing original target curve";return false;}
    const auto minimum=storePitchControlPoints?std::min(0.0,original.front().timeSeconds):0.0;
    const auto maximum=storePitchControlPoints?std::max(note.durationSeconds,original.back().timeSeconds):note.durationSeconds;
    return composePitchStrokeCurve(original,stroke,minimum,maximum,storePitchControlPoints,out,error,preserveLeft,preserveRight);
}

bool composeConnectionPitchStroke(const TrackData& track,const std::vector<NativeConnection>& connections,
    const juce::String& id,const std::vector<PitchCurveEditPoint>& absoluteStroke,std::vector<PitchCurveEditPoint>& out,juce::String& error)
{
    out.clear();error.clear();
    if(std::count_if(connections.begin(),connections.end(),[&](const auto& c){return c.id==id;})!=1)
    {error="Unknown or ambiguous graph stroke target";return false;}
    const auto curves=resolvedConnectionPitchCurves(track,connections);
    const auto current=std::find_if(curves.begin(),curves.end(),[&](const auto& c){return c.connectionId==id;});
    if(current==curves.end()||!current->active){error=current==curves.end()?juce::String("Unresolved graph stroke target"):current->issue;return false;}
    if(absoluteStroke.size()<2)return false;
    auto stroke=absoluteStroke;std::stable_sort(stroke.begin(),stroke.end(),[](const auto& a,const auto& b){return a.timeSeconds<b.timeSeconds;});
    for(std::size_t i=0;i<stroke.size();++i)
    {
        if(!finiteGraphPoint(stroke[i])||stroke[i].targetMidi<0||stroke[i].targetMidi>127||(i&&stroke[i].timeSeconds<=stroke[i-1].timeSeconds))
        {error="Invalid absolute graph stroke controls";return false;}
        stroke[i].shape=PitchCurveShape::linear;
    }
    const auto from=std::max(current->fromSeconds,stroke.front().timeSeconds),to=std::min(current->toSeconds,stroke.back().timeSeconds);
    if(to-from<1.e-6)return false;
    std::vector<PitchCurveEditPoint> clipped{{from,evaluatePitchCurve(stroke,from),PitchCurveShape::linear}};
    for(const auto& p:stroke)if(p.timeSeconds>from&&p.timeSeconds<to)clipped.push_back(p);
    clipped.push_back({to,evaluatePitchCurve(stroke,to),PitchCurveShape::linear});
    // Even an artificially clipped graph edge remains attached to its proposed
    // live note score. If note+graph painting agrees there, the guard has no bump.
    return composePitchStrokeCurve(current->points,clipped,current->fromSeconds,current->toSeconds,true,out,error,true,true);
}

namespace
{
std::vector<PitchPoint> sliceMaterialContour(const NoteData& original,double from,double to,bool preserveTarget)
{
    const auto& points=original.contour;if(points.empty())return {};
    const auto exact = [&](PitchPoint p,double t)
    {
        if(preserveTarget){p.manualTargetCents=renderedPitchCents(original,p);p.hasManualTarget=true;}
        p.timeSeconds=t;return p;
    };
    const auto at = [&](double t)
    {
        auto right=std::upper_bound(points.begin(),points.end(),t,[](double v,const auto& p){return v<p.timeSeconds;});
        if(right==points.begin())return exact(*right,t);
        const auto left=std::prev(right);
        // A detector knot's own voicing is authoritative. Looking ahead at an
        // unvoiced NEXT point here would erase the preceding voiced interval.
        if(left->timeSeconds==t||right==points.end())return exact(*left,t);
        auto p=*left;const auto span=right->timeSeconds-p.timeSeconds;
        const auto u=span>1.e-12?static_cast<float>((t-p.timeSeconds)/span):0.f;
        const auto target=renderedPitchCents(original,*left)
            +(renderedPitchCents(original,*right)-renderedPitchCents(original,*left))*u;
        p.relativeCents+=(right->relativeCents-p.relativeCents)*u;
        p.withoutVibratoCents+=(right->withoutVibratoCents-p.withoutVibratoCents)*u;
        p.manualTargetCents=preserveTarget?target:p.manualTargetCents+(right->manualTargetCents-p.manualTargetCents)*u;
        p.voiced=p.voiced&&right->voiced;p.hasManualTarget=preserveTarget||(p.hasManualTarget&&right->hasManualTarget);
        p.timeSeconds=t;return p;
    };
    std::vector<PitchPoint> result{at(from)};
    // Never drop a nearby real voiced support point merely because a cut is
    // within an epsilon: AND-voicing interpolation makes that acoustically real.
    for(auto p:points)if(p.timeSeconds>from&&p.timeSeconds<to)result.push_back(p);
    result.push_back(at(to));for(auto& p:result)p.timeSeconds-=from;return result;
}
}

ProjectModel::MaterialSourceReplacementReport ProjectModel::replaceNoteMaterialSource(
    const juce::String& noteId,const juce::File& audio,const std::vector<SampleRegionSetting>& completeRows,
    const juce::String& audioId,const juce::String& regionId,const juce::String& chosenAlias,
    juce::String& error,bool apply)
{
    MaterialSourceReplacementReport report;error.clear();
    const auto reject=[&](const juce::String& why)
    {report.rejected=true;report.changed=false;error=why;report.diagnostics.add(why);return report;};
    if(noteId.isEmpty()||audioId.isEmpty()||regionId.isEmpty()||completeRows.empty())return reject("A stable note/audio/pronunciation identity is required.");
    OttoMaterialIO::Document annotation;annotation.audio.push_back({audioId,audio.getFileName(),completeRows});
    if(!OttoMaterialIO::validate(annotation,error))return reject(error);
    const auto rowIt=std::find_if(completeRows.begin(),completeRows.end(),[&](const auto& r){return r.materialAudioId==audioId&&r.materialRegionId==regionId;});
    if(rowIt==completeRows.end())return reject("The chosen source pronunciation no longer exists.");
    const auto& row=*rowIt;const auto alias=chosenAlias.isEmpty()?row.name:chosenAlias;
    if(alias!=row.name&&!row.additionalAliases.contains(alias))return reject("The selected alias does not belong to this pronunciation identity.");
    juce::AudioFormatManager formats;formats.registerBasicFormats();
    std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(audio));
    if(!reader||reader->sampleRate<=0)return reject("The selected physical source audio cannot be read.");
    const auto duration=reader->lengthInSamples/reader->sampleRate;
    for(const auto& r:completeRows)if(r.regionEndSeconds>duration+1.e-8)return reject("A source pronunciation lies beyond the actual recording.");
    const auto fingerprint=juce::SHA256(audio).toHexString();
    constexpr double epsilon=1.e-8;
    const auto near=[](double a,double b){return std::abs(a-b)<=epsilon;};
    bool changed=false;
    {
        const juce::ScopedLock guard(lock);
        TrackData* track=nullptr;ClipData* original=nullptr;NoteData* selected=nullptr;int matches=0;
        for(auto& t:project.tracks)for(auto& c:t.clips)for(auto& n:c.notes)if(n.id==noteId)
        {track=&t;original=&c;selected=&n;++matches;}
        if(matches!=1)return reject("The target note is missing or ambiguous; no source was changed.");
        // Capability, never import history: native physical-file renderers only.
        if(!trackShowsAllNativeRegions(*track)||track->pitchAlgorithm==PitchAlgorithm::utau)
            return reject("This renderer has no explicit physical-recording adapter. Its engine and voicebank were left unchanged.");
        if(selected->materialLocalOverride)return reject("The selected note has a private material override; clear it explicitly before replacing its source.");
        if(selected->durationSeconds<.001)return reject("A source assignment requires a target note at least 1ms long.");
        ClipData* owner=original;
        if(!original->parts.empty())
        {
            owner=nullptr;
            for(auto& part:original->parts)if(part.id==selected->clipPartId){if(owner)return reject("Ambiguous source-child ownership.");owner=&part;}
            if(!owner)return reject("The note has no explicit source-child owner.");
            for(const auto& p:original->parts)if(!p.parts.empty())return reject("Nested physical source owners are unsupported.");
        }
        const auto ownerOffset=original->parts.empty()?0.0:owner->startSeconds;
        auto clockOwner=*owner;
        if(!original->parts.empty())
        {
            clockOwner.notes.clear();
            for(auto n:original->notes)if(n.clipPartId==owner->id){n.startSeconds-=ownerOffset;clockOwner.notes.push_back(std::move(n));}
        }
        const auto actualClock=nativeClipClock(clockOwner);
        const bool sameMappedEdges=!actualClock.empty()
            &&near(owner->sourceOffsetSeconds+nativeSourceTimeAt(actualClock,selected->startSeconds-ownerOffset),row.regionStartSeconds)
            &&near(owner->sourceOffsetSeconds+nativeSourceTimeAt(actualClock,selected->startSeconds-ownerOffset+selected->durationSeconds),row.regionEndSeconds);
        report.clipId=original->id;report.sourcePartId=owner->id;
        const bool samePhysical=sameMappedEdges&&owner->sourceFile==audio&&selected->materialAudioId==audioId&&selected->materialRegionId==regionId
            &&near(selected->nativeSourceStartSeconds,row.regionStartSeconds)&&near(selected->nativeSourceEndSeconds,row.regionEndSeconds)
            &&selected->nativeUnpitched==row.nativeUnpitched
            &&(owner->materialSourceFingerprint.isEmpty()||owner->materialSourceFingerprint==fingerprint);
        const bool followLabel=chosenAlias.isNotEmpty()||selected->materialLabelFollowsAlias;
        const auto display=followLabel?alias:selected->label;
        const bool aliasChanges=display!=selected->label||alias!=(selected->materialBoundAlias.isEmpty()?selected->label:selected->materialBoundAlias)
            ||selected->materialAliasFollowsPrimary!=(alias==row.name)||selected->materialLabelFollowsAlias!=followLabel||selected->materialReferenceUnresolved
            ||selected->pronunciationKind!=row.pronunciationKind||selected->simplifiedConsonant!=row.simplifiedConsonant
            ||selected->vowelSplitsExplicit!=row.vowelSplitsExplicit;
        if(samePhysical)
        {
            report.pendingAnalysis=owner->materialSourceAnalysisPending;
            if(report.pendingAnalysis)report.ticket={noteId,original->id,owner->id,audio,owner->materialSourceFingerprint};
            report.changed=aliasChanges;
            if(apply&&aliasChanges)
            {
                pushUndoLocked();selected->label=display;selected->materialBoundAlias=alias;
                selected->materialAliasFollowsPrimary=alias==row.name;selected->materialLabelFollowsAlias=followLabel;
                selected->materialReferenceUnresolved=false;selected->pronunciationKind=row.pronunciationKind;
                selected->simplifiedConsonant=row.simplifiedConsonant;selected->vowelSplitsExplicit=row.vowelSplitsExplicit;changed=true;
            }
        }
        else
        {
            if(original->glideConnectedFromPrevious||original->glideConnectedToNext)
                return reject("An external physical glide requires a larger score snapshot before source replacement.");
            const auto begin=selected->startSeconds-ownerOffset,end=begin+selected->durationSeconds;
            if(begin<owner->audioStartSeconds-epsilon||end>owner->audioStartSeconds+owner->audioLength()+epsilon)
                return reject("The target note extends outside its physical source-owner domain.");
            auto candidate=*original;
            const auto affected=[&](const NoteData& n){return original->parts.empty()||original->nativeAudioLinked||n.clipPartId==owner->id;};
            bool authored=false,derived=false,neutral=false,nonNeutral=false;
            for(const auto& n:original->notes)if(affected(n))
            {
                authored|=!n.pitchControlPoints.empty();derived|=n.pitchControlPoints.empty();
                neutral|=nativePendingPitchIsNeutral(n);nonNeutral|=!nativePendingPitchIsNeutral(n);
                if(n.nativeIndependentPitch||n.robustPitchCurve)return reject("Independent/robust pitch needs a larger exact score partition.");
            }
            if(authored&&derived)return reject("Mixed authored and derived score authorities require an explicit shared partition.");
            if(neutral&&nonNeutral)return reject("Mixed pending-neutral and expressive source notes must finish analysis first.");
            const auto lines=sharedPitchLines(*track);const auto envelopes=nativeSharedEnvelopes(*track);
            for(auto& n:candidate.notes)if(affected(n))
            {
                const auto old=std::find_if(original->notes.begin(),original->notes.end(),[&](const auto& p){return p.id==n.id;});
                const auto absolute=original->startSeconds+n.startSeconds;
                auto score=ownPitchPoints(*old);for(auto& p:score)p.timeSeconds+=absolute;
                if(const auto* member=lines.memberFor(n.id))
                {
                    if(member->line->pieces.size()!=1)return reject("Piecewise independent pitch transitions cannot yet be rebound exactly.");
                    score=member->line->pieces.front().points;
                }
                if(score.empty()||!std::isfinite(n.startSeconds)||!std::isfinite(n.durationSeconds))return reject("Invalid target score geometry.");
                if(!old->pitchControlPoints.empty())
                {
                    // Keep real authored controls local; do not clone an entire
                    // old shared line into every independently owned source.
                    if(!sliceNativePitchCurve(score,absolute,absolute+n.durationSeconds,n.pitchControlPoints))
                        return reject("A target cubic cannot be represented exactly after source partition.");
                }
                else n.pitchControlPoints.clear();
                std::vector<double> times{0,n.durationSeconds};
                for(const auto& p:old->contour)if(p.timeSeconds>0&&p.timeSeconds<n.durationSeconds)times.push_back(p.timeSeconds);
                if(n.id==noteId)for(double t=0;t<n.durationSeconds;t+=.005)times.push_back(t);
                for(const auto& p:score)if(p.timeSeconds>absolute&&p.timeSeconds<absolute+n.durationSeconds)times.push_back(p.timeSeconds-absolute);
                std::sort(times.begin(),times.end());times.erase(std::unique(times.begin(),times.end()),times.end());
                auto facts=sliceMaterialContour(*old,0,n.durationSeconds,false);
                if(facts.empty())facts={{0,0,0,true},{n.durationSeconds,0,0,true}};
                const auto factAt=[&](double t)
                {
                    auto right=std::lower_bound(facts.begin(),facts.end(),t,[](const auto& p,double time){return p.timeSeconds<time;});
                    if(right==facts.end())return facts.back();
                    if(right==facts.begin()||right->timeSeconds==t)return *right;
                    auto p=*std::prev(right);const auto u=static_cast<float>((t-p.timeSeconds)/(right->timeSeconds-p.timeSeconds));
                    p.relativeCents+=(right->relativeCents-p.relativeCents)*u;p.withoutVibratoCents+=(right->withoutVibratoCents-p.withoutVibratoCents)*u;
                    p.voiced=p.voiced&&right->voiced;return p;
                };
                // Derived acoustic samples stay read-only target supports. For
                // untouched all-neutral notes, retain their PCM-neutral predicate.
                if(n.id==noteId||!nativePendingPitchIsNeutral(*old))
                {
                    std::vector<PitchPoint> frozen;
                    for(auto t:times)
                    {
                        auto p=factAt(t);p.timeSeconds=t;p.hasManualTarget=true;
                        p.manualTargetCents=(evaluatePitchCurve(score,absolute+t)-n.midiNote)*100;
                        if(n.id==noteId){p.relativeCents=p.withoutVibratoCents=0;p.voiced=false;}
                        frozen.push_back(p);
                    }
                    n.contour=std::move(frozen);
                }
                n.materialTargetPitchLocked=true;
                if(const auto e=envelopes.find(n.id);e!=envelopes.end())
                {
                    const auto audible=sliceNativeSharedEnvelope(*e->second.points,absolute,absolute+n.durationSeconds);
                    auto plain=unscaledAmplitudeEnvelope(audible,n.amplitudeEnvelopeBasePercent);
                    const auto check=scaledAmplitudeEnvelope(plain,n.amplitudeEnvelopeBasePercent);
                    for(std::size_t i=0;i<check.size();++i)if(std::abs(check[i].gainDb-audible[i].gainDb)>1.e-5f)
                        return reject("The shared gain would saturate with the unchanged per-note base percent.");
                    n.amplitudeEnvelope=std::move(plain);
                }
            }
            auto newSelected=std::find_if(candidate.notes.begin(),candidate.notes.end(),[&](const auto& n){return n.id==noteId;});
            auto base=*owner;base.parts.clear();base.notes.clear();
            for(auto n:original->notes)if(original->parts.empty()||n.clipPartId==owner->id)
            {
                n.startSeconds-=ownerOffset;
                if(n.id!=noteId&&n.startSeconds<end-epsilon&&n.startSeconds+n.durationSeconds>begin+epsilon)
                    return reject("An overlapping note makes the requested physical source domain ambiguous.");
                base.notes.push_back(std::move(n));
            }
            const auto oldClock=nativeClipClock(base);
            if(oldClock.size()<2)return reject("The original source clock is unavailable.");
            for(std::size_t i=1;i<oldClock.size();++i)if(oldClock[i].targetSeconds<=oldClock[i-1].targetSeconds||oldClock[i].sourceSeconds<=oldClock[i-1].sourceSeconds)
                return reject("The original source clock is not monotonic.");
            base.sourceTimeMap=oldClock;base.startSeconds=0;base.nativeAudioLinked=false;base.nativePitchGroupId.clear();
            if(original->parts.empty())
            {
                // The parent retains these master layers; never bake them into
                // children and then multiply them a second time during expansion.
                base.gain=1;base.muted=owner->materialSourceAnalysisPending;base.materialSourceResumeMuted=false;
                base.gainEnvelope.clear();base.inheritedGainEnvelopes.clear();
                base.fadeInSeconds=base.fadeOutSeconds=base.crossfadeInSeconds=base.crossfadeOutSeconds=0;
            }
            const auto sourceA=nativeSourceTimeAt(oldClock,begin),sourceB=nativeSourceTimeAt(oldClock,end);
            if(sourceB<=sourceA+epsilon)return reject("The selected note has no invertible source interval.");
            std::vector<ClipData> partition;
            const auto crop=[&](double from,double to,const juce::String& id)->bool
            {
                if(to-from<=epsilon)return true;
                auto pieces=slicedClipParts({base},from,to,false,true);if(pieces.size()!=1)return false;
                auto p=std::move(pieces.front());p.id=id;p.startSeconds=ownerOffset+from;p.notes.clear();p.nativeAudioLinked=false;p.nativePitchGroupId.clear();
                if(!clipIsRecording(base))p.sourceFile=juce::File{}; // score/MIDI provenance is not PCM
                partition.push_back(std::move(p));return true;
            };
            const auto leftId=begin>epsilon?(original->parts.empty()?makeId("source-left"):owner->id):juce::String{};
            const auto rightId=end<base.durationSeconds-epsilon?makeId("source-right"):juce::String{};
            if(!crop(0,begin,leftId))return reject("The original left source domain cannot be preserved.");
            const auto selectedId=makeId("source-binding");
            if(!crop(begin,end,selectedId))return reject("The new source domain cannot be represented.");
            auto& replacement=partition.back();replacement.sourceFile=audio;replacement.sourceOffsetSeconds=row.regionStartSeconds;
            replacement.sourceDurationSeconds=row.regionEndSeconds-row.regionStartSeconds;
            replacement.audioStartSeconds=0;replacement.audioDurationSeconds=-1;replacement.durationSeconds=end-begin;
            replacement.sourceTimeMap={{0,0}};
            for(const auto& p:oldClock)if(p.targetSeconds>begin+epsilon&&p.targetSeconds<end-epsilon)
                replacement.sourceTimeMap.push_back({p.targetSeconds-begin,(p.sourceSeconds-sourceA)/(sourceB-sourceA)*replacement.sourceDurationSeconds});
            replacement.sourceTimeMap.push_back({end-begin,replacement.sourceDurationSeconds});
            replacement.nativeSourcePitch.reset();replacement.nativeSourcePitchComplete=false;replacement.nativeSourcePitchPending=true;
            replacement.nativeTrimClock.clear();replacement.materialSourceFingerprint=fingerprint;
            replacement.materialSourceAnalysisPending=true;
            replacement.materialSourceResumeMuted=original->parts.empty()?false
                :owner->materialSourceAnalysisPending?owner->materialSourceResumeMuted:owner->muted;
            replacement.muted=true; // model-only pending render gate, not a fake C4 source
            if(!crop(end,base.durationSeconds,rightId))return reject("The original right source domain cannot be preserved.");
            for(auto& n:candidate.notes)if(original->parts.empty()||n.clipPartId==owner->id)
            {
                if(n.id==noteId)n.clipPartId=selectedId;
                else if(n.startSeconds+n.durationSeconds<=ownerOffset+begin+epsilon)n.clipPartId=leftId;
                else if(n.startSeconds>=ownerOffset+end-epsilon)n.clipPartId=rightId;
                else return reject("A note crosses the exact replacement source cut.");
                if(n.clipPartId.isEmpty())return reject("A retained note lost its explicit source owner.");
            }
            if(original->parts.empty())
            {
                candidate.parts=std::move(partition);
                if(candidate.materialSourceAnalysisPending)candidate.muted=candidate.materialSourceResumeMuted;
                candidate.materialSourceFingerprint.clear();candidate.materialSourceAnalysisPending=false;
                candidate.nativeSourcePitch.reset();candidate.nativeSourcePitchComplete=false;candidate.nativeSourcePitchPending=false;candidate.nativeTrimClock.clear();
            }
            else
            {
                const auto at=std::find_if(candidate.parts.begin(),candidate.parts.end(),[&](const auto& p){return p.id==owner->id;});
                const auto index=static_cast<std::size_t>(at-candidate.parts.begin());candidate.parts.erase(at);
                candidate.parts.insert(candidate.parts.begin()+static_cast<std::ptrdiff_t>(index),partition.begin(),partition.end());
            }
            candidate.nativeAudioLinked=false;
            newSelected=std::find_if(candidate.notes.begin(),candidate.notes.end(),[&](const auto& n){return n.id==noteId;});
            newSelected->sourceMidiCenter=-1;newSelected->sourcePitchMeasured=false;newSelected->nativeUnpitched=row.nativeUnpitched;
            newSelected->label=display;newSelected->materialBoundAlias=alias;newSelected->materialAliasFollowsPrimary=alias==row.name;
            newSelected->materialLabelFollowsAlias=followLabel;newSelected->materialReferenceUnresolved=false;
            newSelected->materialAudioId=audioId;newSelected->materialRegionId=regionId;
            newSelected->materialSourceStartSeconds=newSelected->nativeSourceStartSeconds=row.regionStartSeconds;
            newSelected->materialSourceEndSeconds=newSelected->nativeSourceEndSeconds=row.regionEndSeconds;
            newSelected->materialAdditionalAliases=row.additionalAliases;newSelected->materialConnectedNextRegionId=row.connectedNextRegionId;
            newSelected->pronunciationKind=row.pronunciationKind;newSelected->simplifiedConsonant=row.simplifiedConsonant;newSelected->vowelSplitsExplicit=row.vowelSplitsExplicit;newSelected->nativeRole=row.role;
            newSelected->nativeProvenance=row.provenance;newSelected->nativeConfidence=row.confidence;
            newSelected->nativeSegments=SampleSettings::nativeSegmentsFor(row);
            report.changed=true;report.pendingAnalysis=true;report.sourcePartId=selectedId;
            report.ticket={noteId,original->id,selectedId,audio,fingerprint};
            if(apply){pushUndoLocked();*original=std::move(candidate);changed=true;}
        }
    }
    if(changed)sendChangeMessage();return report;
}

ProjectModel::MaterialUpdateReport ProjectModel::applySourceSettingsWithReport(
    const juce::File& source, const std::vector<SampleRegionSetting>& rows, bool apply)
{
    MaterialUpdateReport report;
    constexpr double edgeTolerance = 1.0e-5; // seconds; legacy matching only
    constexpr double epsilon = 1.0e-8;
    const auto near = [](double a, double b) { return std::abs(a - b) <= epsilon; };
    const auto reject = [&](const juce::String& reason)
    {
        report.rejected = true;
        report.updatedNotes = report.boundLegacyNotes = 0;
        report.diagnostics.add(reason);
    };
    if (source == juce::File{})
    { reject("Material updates require an owning source audio file."); return report; }
    // Duplicate composite IDs are malformed, not a reason to choose the first.
    for (std::size_t i = 0; i < rows.size(); ++i)
    {
        const auto& row = rows[i];
        if (!std::isfinite(row.regionStartSeconds) || !std::isfinite(row.regionEndSeconds)
            || row.regionStartSeconds < 0 || row.regionEndSeconds <= row.regionStartSeconds
            || !std::isfinite(row.confidence))
        { reject("Invalid material source interval."); return report; }
        for (std::size_t j = 0; j < i; ++j)
            if (row.materialAudioId.isNotEmpty() && row.materialRegionId.isNotEmpty()
                && row.materialAudioId == rows[j].materialAudioId
                && row.materialRegionId == rows[j].materialRegionId)
            { reject("Duplicate material audio/region identity."); return report; }
    }
    const auto segmentsEqual = [](const auto& a, const auto& b)
    {
        if (a.size() != b.size()) return false;
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            const auto& x = a[i]; const auto& y = b[i];
            if (x.id != y.id || x.alias != y.alias || x.role != y.role
                || x.sourceStartSeconds != y.sourceStartSeconds || x.sourceEndSeconds != y.sourceEndSeconds
                || x.provenance != y.provenance || x.confidence != y.confidence
                || x.alignmentSeconds != y.alignmentSeconds || x.overlapSeconds != y.overlapSeconds
                || x.stretchable != y.stretchable || x.stretchWeight != y.stretchWeight) return false;
        }
        return true;
    };
    bool changed = false;
    {
        const juce::ScopedLock guard(lock);
        struct Plan { ClipData* destination; ClipData replacement; };
        std::vector<Plan> plans;
        auto candidateConnections = project.nativeConnections;
        bool topologyGraphChanged = false;
        std::vector<TrackData*> topologyTracks;
        double mediaDuration = -1.0;
        juce::String observedSourceFingerprint;
        // Work on only touched clips, not an additional full-project snapshot.
        for (auto& track : project.tracks) for (auto& original : track.clips)
        {
            if (track.accompaniment) continue;
            bool relevant = original.parts.empty() ? original.sourceFile == source
                : std::any_of(original.parts.begin(), original.parts.end(),
                    [&](const auto& part) { return part.sourceFile == source; });
            if (!relevant) continue;
            auto candidate = original;
            bool clipChanged = false;
            const auto updateOwner = [&](ClipData& owner, double noteOffset,
                                          const juce::String& partId) -> bool
            {
                struct Binding { NoteData* note; const SampleRegionSetting* row; double start, end; bool edgesChanged, intact; };
                std::vector<Binding> bindings;
                auto clockOwner = owner;
                if (!candidate.parts.empty())
                {
                    clockOwner.notes.clear();
                    for (auto note : candidate.notes) if (note.clipPartId == partId)
                    { note.startSeconds -= noteOffset; clockOwner.notes.push_back(std::move(note)); }
                }
                const auto clock = nativeClipClock(clockOwner);
                // Detect new-identity topology before ordinary boundary retargeting
                // and before local-override skipping. It needs an exact partition of
                // the OLD target/source coverage, not a stretch onto the retained ID.
                struct Topology { std::size_t index; std::vector<const SampleRegionSetting*> chain; };
                std::vector<Topology> topology;
                const auto placed = [&](const SampleRegionSetting& row)
                {
                    return std::any_of(candidate.notes.begin(),candidate.notes.end(),[&](const auto& n)
                    { return (candidate.parts.empty()||n.clipPartId==partId)
                        &&n.materialAudioId==row.materialAudioId&&n.materialRegionId==row.materialRegionId; });
                };
                const auto topologyFailure = [&](const juce::String& detail)
                { reject("Unsupported material topology: " + detail + ". Publication cancelled; original coverage and musical edits retained."); return false; };
                for(std::size_t ni=0;ni<candidate.notes.size();++ni)
                {
                    const auto& note=candidate.notes[ni];
                    if(!candidate.parts.empty()&&note.clipPartId!=partId)continue;
                    if(note.materialAudioId.isEmpty()||note.materialRegionId.isEmpty())continue;
                    auto retained=std::find_if(rows.begin(),rows.end(),[&](const auto& r)
                    {return r.materialAudioId==note.materialAudioId&&r.materialRegionId==note.materialRegionId;});
                    if(retained==rows.end())continue; // deleted/merged IDs retain unresolved snapshots
                    const auto first=note.materialSourceStartSeconds>=0?note.materialSourceStartSeconds:note.nativeSourceStartSeconds;
                    const auto last=note.materialSourceEndSeconds>=0?note.materialSourceEndSeconds:note.nativeSourceEndSeconds;
                    if(first<0||last<=first)continue;
                    if(retained->regionStartSeconds<=first+epsilon&&retained->regionEndSeconds>=last-epsilon)continue;
                    const bool newCoverage=std::any_of(rows.begin(),rows.end(),[&](const auto& r)
                    {
                        const auto a=std::max(first,r.regionStartSeconds),b=std::min(last,r.regionEndSeconds);
                        return r.materialAudioId==note.materialAudioId&&r.materialRegionId!=note.materialRegionId
                            &&b>a+epsilon&&(a<retained->regionStartSeconds-epsilon||b>retained->regionEndSeconds+epsilon)&&!placed(r);
                    });
                    if(!newCoverage)continue;
                    if(!candidate.parts.empty()||candidate.nativeAudioLinked||!trackShowsAllNativeRegions(track))
                        return topologyFailure("only an intact single-source native owner can be partitioned");
                    if(note.materialLocalOverride)return topologyFailure("local material override at "+note.id);
                    if(!near(retained->regionStartSeconds,first))return topologyFailure("retained ID must be the first chain member");
                    if(!near(owner.sourceOffsetSeconds+nativeSourceTimeAt(clock,note.startSeconds),first)
                        ||!near(owner.sourceOffsetSeconds+nativeSourceTimeAt(clock,note.startSeconds+note.durationSeconds),last))
                        return topologyFailure("cropped or independently retimed pronunciation "+note.id);
                    Topology expansion{ni,{&*retained}};
                    while(expansion.chain.back()->regionEndSeconds<last-epsilon)
                    {
                        const auto* previous=expansion.chain.back();
                        auto next=std::find_if(rows.begin(),rows.end(),[&](const auto& r)
                        {return r.materialAudioId==note.materialAudioId&&r.materialRegionId==previous->connectedNextRegionId;});
                        if(next==rows.end()||next->materialRegionId.isEmpty()||placed(*next)
                            ||!near(next->regionStartSeconds,previous->regionEndSeconds)||next->regionEndSeconds>last+epsilon
                            ||std::find(expansion.chain.begin(),expansion.chain.end(),&*next)!=expansion.chain.end())
                            return topologyFailure("new IDs must form an explicit contiguous chain filling the previous source interval");
                        expansion.chain.push_back(&*next);
                    }
                    if(expansion.chain.size()<2||!near(expansion.chain.back()->regionEndSeconds,last))
                        return topologyFailure("chain does not preserve the complete previous source interval");
                    for(const auto& r:rows)
                        if(r.materialAudioId==note.materialAudioId&&r.regionStartSeconds<last-epsilon&&r.regionEndSeconds>first+epsilon
                            &&!placed(r)&&std::find(expansion.chain.begin(),expansion.chain.end(),&r)==expansion.chain.end())
                            return topologyFailure("ambiguous overlapping new pronunciation IDs");
                    topology.push_back(std::move(expansion));
                }
                if(!topology.empty())
                {
                    if(clock.size()<2)return topologyFailure("missing source clock");
                    for(std::size_t i=1;i<clock.size();++i)
                        if(clock[i].targetSeconds<=clock[i-1].targetSeconds+epsilon||clock[i].sourceSeconds<=clock[i-1].sourceSeconds+epsilon)
                            return topologyFailure("non-monotonic source clock");
                    for(std::size_t i=0;i<owner.sourceTimeMap.size();++i)
                    {
                        const auto& p=owner.sourceTimeMap[i];
                        if(!std::isfinite(p.targetSeconds)||!std::isfinite(p.sourceSeconds)
                            ||p.targetSeconds<owner.audioStartSeconds-epsilon||p.targetSeconds>owner.audioStartSeconds+owner.audioLength()+epsilon
                            ||p.sourceSeconds<0||p.sourceSeconds>owner.sourceDurationSeconds+epsilon
                            ||(i&&(p.targetSeconds<=owner.sourceTimeMap[i-1].targetSeconds+epsilon||p.sourceSeconds<=owner.sourceTimeMap[i-1].sourceSeconds+epsilon)))
                            return topologyFailure("existing warp would require clamping");
                    }
                    if(mediaDuration<0)
                    {
                        juce::AudioFormatManager formats;formats.registerBasicFormats();
                        std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(source));
                        if(!reader||reader->sampleRate<=0)return topologyFailure("cannot validate source media");
                        mediaDuration=reader->lengthInSamples/reader->sampleRate;
                    }
                    if(owner.sourceOffsetSeconds<0||owner.sourceOffsetSeconds+clock.back().sourceSeconds>mediaDuration+epsilon)
                        return topologyFailure("source interval outside recording");
                    std::vector<std::size_t> ordered;
                    for(std::size_t i=0;i<candidate.notes.size();++i)ordered.push_back(i);
                    std::stable_sort(ordered.begin(),ordered.end(),[&](auto a,auto b){return candidate.notes[a].startSeconds<candidate.notes[b].startSeconds;});
                    const bool automaticOwner=std::all_of(candidate.notes.begin(),candidate.notes.end(),[](const auto& n){return n.pitchControlPoints.empty();});
                    if(!automaticOwner&&std::any_of(candidate.notes.begin(),candidate.notes.end(),[](const auto& n){return n.pitchControlPoints.empty();}))
                        return topologyFailure("mixed authored and derived pitch requires a larger shared automation partition");
                    if(automaticOwner&&std::any_of(candidate.notes.begin(),candidate.notes.end(),[](const auto& n){return n.contour.size()<2;}))
                        return topologyFailure("derived pitch requires its acoustic contour before partition");
                    const bool neutralOwner=std::all_of(candidate.notes.begin(),candidate.notes.end(),nativePendingPitchIsNeutral);
                    if(!neutralOwner&&std::any_of(candidate.notes.begin(),candidate.notes.end(),nativePendingPitchIsNeutral))
                        return topologyFailure("mixed pending-neutral source and expressive members must finish acoustic analysis first");
                    double cursor=owner.audioStartSeconds;
                    for(auto index:ordered)
                    {
                        const auto& n=candidate.notes[index];
                        if(n.materialLocalOverride||n.nativeIndependentPitch||n.nativeUnpitched||n.robustPitchCurve||!near(n.startSeconds,cursor)
                            ||n.nativeEnvelope.mode!=0||n.utauTailFadeMode!=0||!n.utauFlagCurves.empty()
                            ||!n.diffSingerPitchOffset.empty()||!n.diffSingerPitchReference.empty()||!n.cvvcEnvelopes.empty())
                            return topologyFailure("requires contiguous voiced shared notes without local/independent/advanced automation layers");
                        cursor=n.startSeconds+n.durationSeconds;
                        const bool expanded=std::any_of(topology.begin(),topology.end(),[&](const auto& t){return t.index==index;});
                        if(!expanded)
                        {
                            auto r=std::find_if(rows.begin(),rows.end(),[&](const auto& r){return r.materialAudioId==n.materialAudioId&&r.materialRegionId==n.materialRegionId;});
                            if(r!=rows.end()&&(!near(r->regionStartSeconds,n.materialSourceStartSeconds)||!near(r->regionEndSeconds,n.materialSourceEndSeconds)))
                                return topologyFailure("simultaneous neighbor boundary changes need a separate transaction");
                        }
                    }
                    if(!near(cursor,owner.audioStartSeconds+owner.audioLength()))return topologyFailure("notes do not cover the complete audible owner");
                    const auto lines=sharedPitchLines(track);
                    const auto envelopes=nativeSharedEnvelopes(track);
                    const auto& firstNote=candidate.notes[ordered.front()];
                    std::vector<PitchCurveEditPoint> pitch;
                    if(const auto* member=lines.memberFor(firstNote.id))
                    {
                        if(member->line->pieces.size()!=1)return topologyFailure("piecewise independent pitch authority");
                        pitch=member->line->pieces.front().points;
                    }
                    else
                    {
                        if(ordered.size()!=1)return topologyFailure("missing shared pitch authority");
                        pitch=ownPitchPoints(firstNote);for(auto& p:pitch)p.timeSeconds+=candidate.startSeconds+firstNote.startSeconds;
                    }
                    std::vector<AmplitudeEnvelopePoint> envelope;
                    if(const auto found=envelopes.find(firstNote.id);found!=envelopes.end())envelope=*found->second.points;
                    else
                    {
                        if(ordered.size()!=1)return topologyFailure("missing shared gain authority");
                        envelope=scaledAmplitudeEnvelope(firstNote.amplitudeEnvelope,firstNote.amplitudeEnvelopeBasePercent);
                        if(envelope.empty())envelope={{0,scaledEnvelopeGainDb(0,firstNote.amplitudeEnvelopeBasePercent/100.0f)},
                            {firstNote.durationSeconds,scaledEnvelopeGainDb(0,firstNote.amplitudeEnvelopeBasePercent/100.0f)}};
                        for(auto& p:envelope)p.timeSeconds+=candidate.startSeconds+firstNote.startSeconds;
                    }
                    const auto groupStart=candidate.startSeconds+owner.audioStartSeconds;
                    const auto groupEnd=groupStart+owner.audioLength();
                    if(pitch.empty()||pitch.front().timeSeconds<groupStart-epsilon||pitch.back().timeSeconds>groupEnd+epsilon
                        ||envelope.empty()||envelope.front().timeSeconds<groupStart-epsilon||envelope.back().timeSeconds>groupEnd+epsilon)
                        return topologyFailure("external support handles outside the owner require a larger shared partition");
                    std::vector<NoteData> members;
                    std::vector<std::pair<juce::String,std::vector<juce::String>>> chains;
                    for(auto index:ordered)
                    {
                        const auto originalNote=candidate.notes[index];
                        const auto expansion=std::find_if(topology.begin(),topology.end(),[&](const auto& t){return t.index==index;});
                        std::vector<const SampleRegionSetting*> chain;
                        if(expansion!=topology.end())chain=expansion->chain;else chain={nullptr};
                        std::vector<juce::String> ids;
                        for(std::size_t ci=0;ci<chain.size();++ci)
                        {
                            const auto* row=chain[ci];auto n=originalNote;
                            const auto start=row?nativeTargetTimeAt(clock,row->regionStartSeconds-owner.sourceOffsetSeconds):originalNote.startSeconds;
                            const auto end=row?nativeTargetTimeAt(clock,row->regionEndSeconds-owner.sourceOffsetSeconds):originalNote.startSeconds+originalNote.durationSeconds;
                            if(end-start<.001)return topologyFailure("new member shorter than 1ms on the target clock");
                            const auto crop=start-originalNote.startSeconds;
                            n.id=ci==0?originalNote.id:makeId("note");n.startSeconds=start;n.durationSeconds=end-start;
                            n.consonantSeconds=std::max(0.0,std::min(originalNote.consonantSeconds-crop,n.durationSeconds));
                            n.contour=sliceMaterialContour(originalNote,crop,crop+n.durationSeconds,automaticOwner&&!neutralOwner);
                            if(!sliceNativePitchCurve(pitch,candidate.startSeconds+start,candidate.startSeconds+end,n.pitchControlPoints))
                                return topologyFailure("singular or invalid cubic pitch crop");
                            // Derived 5ms acoustic data stays a contour, not hundreds
                            // of newly writable pitch handles. All derived target
                            // segments are linear; cropped target-cents supports keep
                            // drift/modulation exact without promoting source samples.
                            // Neutral stubs additionally retain their PCM predicate.
                            if(automaticOwner)n.pitchControlPoints.clear();
                            n.nativePitchHandlesPlaced=automaticOwner?originalNote.nativePitchHandlesPlaced:true;
                            const auto audibleEnvelope=sliceNativeSharedEnvelope(envelope,candidate.startSeconds+start,candidate.startSeconds+end);
                            n.amplitudeEnvelope=unscaledAmplitudeEnvelope(audibleEnvelope,n.amplitudeEnvelopeBasePercent);
                            const auto roundTripEnvelope=scaledAmplitudeEnvelope(n.amplitudeEnvelope,n.amplitudeEnvelopeBasePercent);
                            for(std::size_t ei=0;ei<audibleEnvelope.size();++ei)
                                if(std::abs(roundTripEnvelope[ei].gainDb-audibleEnvelope[ei].gainDb)>1.e-5f)
                                    return topologyFailure("shared gain cannot be represented with the unchanged per-note base percent without saturation");
                            n.vibratoReferenceDurationSeconds=originalNote.vibratoReferenceDurationSeconds>0?originalNote.vibratoReferenceDurationSeconds:originalNote.durationSeconds;
                            n.vibratoTimeOffsetSeconds=originalNote.vibratoTimeOffsetSeconds+crop;
                            n.sibilantMarkers.clear();for(auto t:originalNote.sibilantMarkers)if(t>=crop&&t<=crop+n.durationSeconds)n.sibilantMarkers.push_back(t-crop);
                            if(row)
                            {
                                if(row->nativeUnpitched)return topologyFailure("split changes voicing classification");
                                const auto alias=materialAliasProjection(n,*row);n.label=alias.label;n.materialBoundAlias=alias.bound;
                                if(alias.missing)report.diagnostics.add("Selected material alias was removed; snapshot retained: "+n.id+" / "+alias.bound);
                                n.nativeRole=row->role;
                                n.nativeProvenance=row->provenance;n.nativeConfidence=row->confidence;
                                n.materialAudioId=row->materialAudioId;n.materialRegionId=row->materialRegionId;
                                n.materialSourceStartSeconds=n.nativeSourceStartSeconds=row->regionStartSeconds;
                                n.materialSourceEndSeconds=n.nativeSourceEndSeconds=row->regionEndSeconds;
                                n.materialAdditionalAliases=row->additionalAliases;n.materialConnectedNextRegionId=row->connectedNextRegionId;
                                n.pronunciationKind=row->pronunciationKind;n.simplifiedConsonant=row->simplifiedConsonant;n.vowelSplitsExplicit=row->vowelSplitsExplicit;n.materialReferenceUnresolved=false;
                                n.nativeSegments=SampleSettings::nativeSegmentsFor(*row);
                            }
                            ids.push_back(n.id);members.push_back(std::move(n));
                        }
                        if(expansion!=topology.end())chains.push_back({originalNote.id,std::move(ids)});
                    }
                    // Shared-line deduplication takes the RIGHT seam handle. Its
                    // incoming cubic must be the LEFT slice's outgoing descriptor.
                    for(std::size_t i=1;i<members.size();++i)
                    {
                        if(members[i-1].pitchControlPoints.empty()||members[i].pitchControlPoints.empty())continue;
                        const auto incoming=members[i-1].pitchControlPoints.back();auto& seam=members[i].pitchControlPoints.front();
                        seam.shape=incoming.shape;seam.bezierX1=incoming.bezierX1;seam.bezierY1=incoming.bezierY1;
                        seam.bezierX2=incoming.bezierX2;seam.bezierY2=incoming.bezierY2;seam.highPrecisionBezier=incoming.highPrecisionBezier;
                    }
                    materializeLegacyConnectionPairs(candidateConnections,orderedConnectionEndpoints(track));
                    for(const auto& chain:chains)for(auto& c:candidateConnections)
                        if(c.leftNoteId==chain.first)c.leftNoteId=chain.second.back(); // incoming remains original first ID
                    candidate.notes=std::move(members);
                    for(const auto& chain:chains)for(std::size_t i=1;i<chain.second.size();++i)
                    {
                        auto left=std::find_if(candidate.notes.begin(),candidate.notes.end(),[&](const auto& n){return n.id==chain.second[i-1];});
                        auto right=std::find_if(candidate.notes.begin(),candidate.notes.end(),[&](const auto& n){return n.id==chain.second[i];});
                        ConnectionEndpoint a{&*left,&candidate,candidate.startSeconds+left->startSeconds};
                        ConnectionEndpoint b{&*right,&candidate,candidate.startSeconds+right->startSeconds};
                        mutateNativeConnectionPair(candidateConnections,a,b,true);
                        for(auto& c:candidateConnections)if(c.leftNoteId==left->id&&c.rightNoteId==right->id)c.boundarySeconds=b.start;
                    }
                    // Freeze an implicit consonant clock before note topology can
                    // alter it. Explicit maps remain byte-for-byte unchanged.
                    if(owner.sourceTimeMap.empty())owner.sourceTimeMap=clock;
                    topologyGraphChanged=true;clipChanged=true;report.updatedNotes+=static_cast<int>(candidate.notes.size());
                    if(std::find(topologyTracks.begin(),topologyTracks.end(),&track)==topologyTracks.end())topologyTracks.push_back(&track);
                }
                bool boundaryChanged = false;
                for (auto& note : candidate.notes)
                {
                    if (candidate.parts.size() && note.clipPartId != partId) continue;
                    const auto start = owner.sourceOffsetSeconds + nativeSourceTimeAt(clock, note.startSeconds - noteOffset);
                    const auto end = owner.sourceOffsetSeconds + nativeSourceTimeAt(clock, note.startSeconds - noteOffset + note.durationSeconds);
                    const SampleRegionSetting* match = nullptr;
                    if (note.materialLocalOverride)
                    {
                        ++report.skippedLocalOverrides;
                        // A local timing/tuning snapshot does not pin its display
                        // alias. Refresh identity text only, using exact variant IDs.
                        for (const auto& row : rows)
                            if (note.materialAudioId.isNotEmpty()&&note.materialRegionId.isNotEmpty()
                                &&row.materialAudioId==note.materialAudioId&&row.materialRegionId==note.materialRegionId)
                            {
                                const auto alias=materialAliasProjection(note,row);
                                if (note.label!=alias.label||note.materialBoundAlias!=alias.bound
                                    ||note.materialLabelFollowsAlias!=alias.labelFollowsAlias
                                    ||note.materialAdditionalAliases!=row.additionalAliases)
                                {
                                    note.label=alias.label;note.materialBoundAlias=alias.bound;
                                    note.materialLabelFollowsAlias=alias.labelFollowsAlias;
                                    note.materialAdditionalAliases=row.additionalAliases;
                                    clipChanged=true;++report.updatedNotes;
                                }
                                break;
                            }
                    }
                    else if (note.materialAudioId.isNotEmpty() || note.materialRegionId.isNotEmpty())
                    {
                        for (const auto& row : rows)
                            if (row.materialAudioId.isNotEmpty() && row.materialRegionId.isNotEmpty()
                                && row.materialAudioId == note.materialAudioId && row.materialRegionId == note.materialRegionId)
                                match = &row;
                        if (!match)
                        {
                            ++report.unresolvedNotes;
                            report.diagnostics.add("Unresolved material reference retained: " + note.id);
                            if (!note.materialReferenceUnresolved)
                            { note.materialReferenceUnresolved = true; clipChanged = true; ++report.updatedNotes; }
                        }
                    }
                    else if (note.materialAudioId.isEmpty() && note.materialRegionId.isEmpty())
                    {
                        int matches = 0;
                        const auto oldStart = note.nativeSourceStartSeconds >= 0 ? note.nativeSourceStartSeconds : start;
                        const auto oldEnd = note.nativeSourceEndSeconds >= 0 ? note.nativeSourceEndSeconds : end;
                        for (const auto& row : rows)
                            if (std::abs(row.regionStartSeconds - oldStart) <= edgeTolerance
                                && std::abs(row.regionEndSeconds - oldEnd) <= edgeTolerance)
                            { ++matches; match = &row; }
                        if (matches != 1)
                        {
                            match = nullptr;
                            if (matches > 1)
                            {
                                ++report.ambiguousLegacyNotes;
                                report.diagnostics.add("Ambiguous legacy source edges; note left unbound: " + note.id);
                            }
                        }
                        if (match && (match->materialAudioId.isEmpty() || match->materialRegionId.isEmpty())) match = nullptr;
                        if (match) ++report.boundLegacyNotes;
                    }
                    const auto baselineStart = note.materialSourceStartSeconds >= 0 ? note.materialSourceStartSeconds
                        : note.nativeSourceStartSeconds >= 0 ? note.nativeSourceStartSeconds : start;
                    const auto baselineEnd = note.materialSourceEndSeconds >= 0 ? note.materialSourceEndSeconds
                        : note.nativeSourceEndSeconds >= 0 ? note.nativeSourceEndSeconds : end;
                    const bool intact = near(start, baselineStart) && near(end, baselineEnd);
                    const bool edgesChanged = match && (!near(baselineStart, match->regionStartSeconds)
                        || !near(baselineEnd, match->regionEndSeconds));
                    // A split/cropped placement still references the pronunciation,
                    // but its subrange cannot be expanded to the whole row safely.
                    if (edgesChanged && !intact)
                    { reject("Boundary update requires an intact source-region placement: " + note.id); return false; }
                    boundaryChanged |= edgesChanged;
                    bindings.push_back({ &note, match, start, end, edgesChanged, intact });
                }
                if (boundaryChanged)
                {
                    if (clock.size() < 2) { reject("Missing source clock."); return false; }
                    for (std::size_t i = 0; i < owner.sourceTimeMap.size(); ++i)
                    {
                        const auto& point = owner.sourceTimeMap[i];
                        const auto sourceLength = owner.sourceDurationSeconds > epsilon
                            ? owner.sourceDurationSeconds : owner.audioLength();
                        if (!std::isfinite(point.targetSeconds) || !std::isfinite(point.sourceSeconds)
                            || point.targetSeconds < owner.audioStartSeconds - epsilon
                            || point.targetSeconds > owner.audioStartSeconds + owner.audioLength() + epsilon
                            || point.sourceSeconds < -epsilon || point.sourceSeconds > sourceLength + epsilon
                            || (i && (point.targetSeconds <= owner.sourceTimeMap[i-1].targetSeconds + epsilon
                                || point.sourceSeconds <= owner.sourceTimeMap[i-1].sourceSeconds + epsilon)))
                        { reject("Existing non-monotonic source clock cannot be updated safely."); return false; }
                    }
                    if(owner.materialSourceFingerprint.isNotEmpty())
                    {
                        if(observedSourceFingerprint.isEmpty())observedSourceFingerprint=juce::SHA256(source).toHexString();
                        if(observedSourceFingerprint!=owner.materialSourceFingerprint)
                        {reject("The explicitly bound recording changed; reselect it before publishing new source boundaries.");return false;}
                    }
                    // Source knots describe ONE coherent transform for this owner.
                    // Local/unresolved/unbound notes pin their actual source edges.
                    std::vector<SourceTimePoint> knots;
                    for (const auto& binding : bindings)
                    {
                        const auto* row = binding.row;
                        const bool rowEdgesChanged = binding.edgesChanged;
                        knots.push_back({ binding.start, rowEdgesChanged ? row->regionStartSeconds : binding.start });
                        knots.push_back({ binding.end, rowEdgesChanged ? row->regionEndSeconds : binding.end });
                    }
                    const auto oldFirst = owner.sourceOffsetSeconds + clock.front().sourceSeconds;
                    const auto oldLast = owner.sourceOffsetSeconds + clock.back().sourceSeconds;
                    const auto pinOuter = [&](double value)
                    {
                        if (std::none_of(knots.begin(), knots.end(), [&](const auto& p) { return near(p.targetSeconds, value); }))
                            knots.push_back({value, value});
                    };
                    pinOuter(oldFirst); pinOuter(oldLast);
                    std::stable_sort(knots.begin(), knots.end(), [](const auto& a, const auto& b) { return a.targetSeconds < b.targetSeconds; });
                    std::vector<SourceTimePoint> unique;
                    for (const auto& knot : knots)
                    {
                        if (!unique.empty() && near(knot.targetSeconds, unique.back().targetSeconds))
                        {
                            if (!near(knot.sourceSeconds, unique.back().sourceSeconds))
                            { reject("Conflicting shared source boundary (including local overrides)."); return false; }
                        }
                        else unique.push_back(knot);
                    }
                    for (std::size_t i = 1; i < unique.size(); ++i)
                        if (unique[i].sourceSeconds <= unique[i-1].sourceSeconds + epsilon)
                        { reject("Source boundaries overlap or reverse; no placements updated."); return false; }
                    // Overlapping target notes cannot independently own one shared clock.
                    for (std::size_t i = 0; i < bindings.size(); ++i)
                        for (std::size_t j = 0; j < i; ++j)
                            if (std::min(bindings[i].end, bindings[j].end)
                                > std::max(bindings[i].start, bindings[j].start) + epsilon)
                            { reject("Boundary edits of overlapping placements in one source owner are unsupported."); return false; }
                    const auto transform = [&](double s) { return nativeSourceTimeAt(unique, s); };
                    const auto newFirst = transform(oldFirst), newLast = transform(oldLast);
                    if (mediaDuration < 0)
                    {
                        juce::AudioFormatManager formats; formats.registerBasicFormats();
                        std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(source));
                        if (!reader || reader->sampleRate <= 0)
                        { reject("Cannot validate source media length for boundary edit."); return false; }
                        mediaDuration = reader->lengthInSamples / reader->sampleRate;
                    }
                    if (newFirst < 0 || newLast > mediaDuration + epsilon)
                    { reject("Boundary outside source media."); return false; }
                    auto mapped = clock;
                    for (const auto& knot : unique)
                        if (knot.targetSeconds > oldFirst + epsilon && knot.targetSeconds < oldLast - epsilon)
                            mapped.push_back({nativeTargetTimeAt(clock, knot.targetSeconds - owner.sourceOffsetSeconds),
                                              knot.targetSeconds - owner.sourceOffsetSeconds});
                    std::stable_sort(mapped.begin(), mapped.end(), [](const auto& a, const auto& b) { return a.targetSeconds < b.targetSeconds; });
                    std::vector<SourceTimePoint> newClock;
                    for (auto point : mapped)
                    {
                        point.sourceSeconds = transform(owner.sourceOffsetSeconds + point.sourceSeconds) - newFirst;
                        if (newClock.empty() || point.targetSeconds > newClock.back().targetSeconds + epsilon)
                            newClock.push_back(point);
                    }
                    juce::String clockError;
                    if(!publishMaterialSourceClock(owner,clock,std::move(newClock),newFirst,newLast,clockError))
                    {reject(clockError);return false;}
                    // Explicitly rebound notes have an independent locked score.
                    // Their measured SOURCE F0 must follow the new source anchors,
                    // not remain sampled from the previous pronunciation edges.
                    if(owner.nativeSourcePitch&&owner.nativeSourcePitchComplete&&!owner.materialSourceAnalysisPending)
                        for(const auto& binding:bindings)if(binding.note->materialTargetPitchLocked)
                        {
                            auto local=*binding.note;local.startSeconds-=noteOffset;refreshNativeTrimPitch(local,owner);
                            binding.note->contour=std::move(local.contour);binding.note->sourceMidiCenter=local.sourceMidiCenter;
                            binding.note->sourcePitchMeasured=local.sourcePitchMeasured;
                        }
                    clipChanged = true;
                }
                for (const auto& binding : bindings)
                {
                    if (!binding.row) continue;
                    auto& note = *binding.note; const auto& row = *binding.row;
                    const auto alias=materialAliasProjection(note,row);
                    const auto& label=alias.label;
                    if(alias.missing)report.diagnostics.add("Selected material alias was removed; source and display snapshot retained: "+note.id+" / "+alias.bound);
                    const auto segments = binding.intact ? SampleSettings::nativeSegmentsFor(row) : note.nativeSegments;
                    // All target-side musical edits remain untouched, even for a
                    // freshly bound legacy note. These are source snapshots only.
                    const bool different = note.label != label || alias.bound!=alias.previousEffective || note.nativeRole != row.role
                        || note.nativeProvenance != row.provenance || note.nativeConfidence != row.confidence
                        || note.materialAudioId != row.materialAudioId || note.materialRegionId != row.materialRegionId
                        || note.materialAdditionalAliases != row.additionalAliases
                        || note.materialConnectedNextRegionId != row.connectedNextRegionId
                        || note.pronunciationKind!=row.pronunciationKind || note.simplifiedConsonant!=row.simplifiedConsonant
                        || note.vowelSplitsExplicit!=row.vowelSplitsExplicit
                        || note.materialReferenceUnresolved || !near(note.materialSourceStartSeconds, row.regionStartSeconds)
                        || !near(note.materialSourceEndSeconds, row.regionEndSeconds)
                        || (binding.intact && (!near(note.nativeSourceStartSeconds, row.regionStartSeconds)
                            || !near(note.nativeSourceEndSeconds, row.regionEndSeconds)))
                        || !segmentsEqual(note.nativeSegments, segments);
                    if (!different) continue;
                    applyMaterialAnnotationSnapshot(note,row,binding.intact,segments);
                    ++report.updatedNotes; clipChanged = true;
                }
                return true;
            };
            if (candidate.parts.empty())
            { if (!updateOwner(candidate, 0, {})) return report; }
            else
            {
                // No nearest-child fallback: explicit ownership is mandatory.
                for (const auto& note : candidate.notes)
                    if (std::none_of(candidate.parts.begin(), candidate.parts.end(),
                        [&](const auto& part) { return part.id == note.clipPartId; }))
                    { reject("Merged placement lacks explicit source ownership."); return report; }
                for (auto& part : candidate.parts) if (part.sourceFile == source)
                {
                    if (!part.parts.empty()) { reject("Nested source owners are unsupported."); return report; }
                    if (!updateOwner(part, part.startSeconds, part.id)) return report;
                }
            }
            if (clipChanged) plans.push_back({ &original, std::move(candidate) });
        }
        if(topologyGraphChanged)
        {
            // Project the directional authority on a shadow track containing ALL
            // prepared placements. Never mutate live neighbor flags in preview.
            for(auto& track:project.tracks)
            {
                if(std::find(topologyTracks.begin(),topologyTracks.end(),&track)==topologyTracks.end())continue;
                bool affected=false;
                auto shadow=track;
                for(std::size_t i=0;i<track.clips.size();++i)
                    if(const auto p=std::find_if(plans.begin(),plans.end(),[&](const auto& p){return p.destination==&track.clips[i];});p!=plans.end())
                    {shadow.clips[i]=p->replacement;affected=true;}
                if(!affected)continue;
                projectConnectionFlags(candidateConnections,orderedConnectionEndpoints(shadow),true);
                for(std::size_t i=0;i<track.clips.size();++i)
                {
                    auto p=std::find_if(plans.begin(),plans.end(),[&](const auto& p){return p.destination==&track.clips[i];});
                    if(p!=plans.end()){p->replacement=std::move(shadow.clips[i]);continue;}
                    const auto& a=track.clips[i];const auto& b=shadow.clips[i];
                    bool flags=a.glideConnectedToNext!=b.glideConnectedToNext||a.glideConnectedFromPrevious!=b.glideConnectedFromPrevious;
                    for(std::size_t n=0;n<a.notes.size();++n)flags|=a.notes[n].connectedToNext!=b.notes[n].connectedToNext||a.notes[n].connectedToPrevious!=b.notes[n].connectedToPrevious;
                    if(flags)plans.push_back({&track.clips[i],std::move(shadow.clips[i])});
                }
            }
        }
        if (apply && !plans.empty())
        {
            pushUndoLocked();
            for (auto& plan : plans) *plan.destination = std::move(plan.replacement);
            if(topologyGraphChanged)project.nativeConnections=std::move(candidateConnections);
            changed = true;
        }
    }
    if (changed) sendChangeMessage();
    return report;
}

juce::ValueTree ProjectModel::toValueTree(const juce::File& projectFile, bool includeDerived) const
{
    const auto data = snapshot();
    juce::ValueTree root("HachiShifterProject");
    root.setProperty("version", 19, nullptr);
    if(data.hamoodState.isNotEmpty())root.setProperty("hamoodState",data.hamoodState,nullptr);
    root.setProperty("nativeSchemaVersion", 2, nullptr);
    root.setProperty("materialAnnotationVersion", 1, nullptr);
    root.setProperty("diffSingerPitchReferenceVersion", 1, nullptr);
    root.setProperty("cvvcEnvelopeVersion", 1, nullptr);
    root.setProperty("name", data.name, nullptr);
    root.setProperty("bpm", data.bpm, nullptr);
    root.setProperty("beatOriginSeconds", data.beatOriginSeconds, nullptr);
    root.setProperty("numerator", data.numerator, nullptr);
    root.setProperty("denominator", data.denominator, nullptr);
    root.setProperty("gridDivision", data.gridDivision, nullptr);
    root.setProperty("noteEditDivision", data.noteEditDivision, nullptr);
    root.setProperty("baseScale", data.baseScale, nullptr);
    root.setProperty("assetManagerVisible", data.assetManagerVisible, nullptr);

    for (const auto& change : data.tempoChanges)
    {
        juce::ValueTree tempoTree("TempoChange");
        tempoTree.setProperty("quarterPosition", change.quarterPosition, nullptr);
        tempoTree.setProperty("bpm", change.bpm, nullptr);
        root.addChild(tempoTree, -1, nullptr);
    }

    for (const auto& connection : data.nativeConnections)
    {
        juce::ValueTree connectionTree("NativeConnection");
        connectionTree.setProperty("id", connection.id, nullptr);
        connectionTree.setProperty("leftNoteId", connection.leftNoteId, nullptr);
        connectionTree.setProperty("rightNoteId", connection.rightNoteId, nullptr);
        connectionTree.setProperty("type", connection.type, nullptr);
        connectionTree.setProperty("boundarySeconds", connection.boundarySeconds, nullptr);
        if(connection.pitchCurveEndpointRelative)
        {
            connectionTree.setProperty("pitchCurveEndpointRelative",true,nullptr);
            connectionTree.setProperty("pitchStartOffsetSeconds",connection.pitchStartOffsetSeconds,nullptr);
            connectionTree.setProperty("pitchEndOffsetSeconds",connection.pitchEndOffsetSeconds,nullptr);
        }
        for (const auto& point : connection.pitchCurve)
        {
            juce::ValueTree pointTree("PitchCurvePoint");
            if(point.highPrecisionBezier)pointTree.setProperty("highPrecisionBezier",true,nullptr);
            pointTree.setProperty("timeSeconds", point.timeSeconds, nullptr);
            pointTree.setProperty("targetMidi", point.targetMidi, nullptr);
            pointTree.setProperty("shape", pitchCurveShapeName(point.shape), nullptr);
            pointTree.setProperty("bezierX1", point.bezierX1, nullptr);
            pointTree.setProperty("bezierY1", point.bezierY1, nullptr);
            pointTree.setProperty("bezierX2", point.bezierX2, nullptr);
            pointTree.setProperty("bezierY2", point.bezierY2, nullptr);
            connectionTree.addChild(pointTree, -1, nullptr);
        }
        for (const auto& point : connection.amplitudeCurve)
        {
            juce::ValueTree pointTree("AmplitudeCurvePoint");
            pointTree.setProperty("timeSeconds", point.timeSeconds, nullptr);
            pointTree.setProperty("gainDb", point.gainDb, nullptr);
            connectionTree.addChild(pointTree, -1, nullptr);
        }
        root.addChild(connectionTree, -1, nullptr);
    }

    for (const auto& track : data.tracks)
    {
        juce::ValueTree trackTree("Track");
        trackTree.setProperty("id", track.id, nullptr);
        trackTree.setProperty("name", track.name, nullptr);
        trackTree.setProperty("compose", track.compose && !track.accompaniment, nullptr);
        trackTree.setProperty("accompaniment", track.accompaniment, nullptr);
        trackTree.setProperty("muted", track.muted, nullptr);
        trackTree.setProperty("solo", track.solo, nullptr);
        trackTree.setProperty("referenceOnly", track.referenceOnly, nullptr);
        trackTree.setProperty("volume", track.volume, nullptr);
        trackTree.setProperty("pan", track.pan, nullptr);
        trackTree.setProperty("smoothOverlaps", track.smoothOverlaps, nullptr);
        trackTree.setProperty("allowNativeAudioOverlap",track.allowNativeAudioOverlap,nullptr);
        trackTree.setProperty("normalizeVolume", track.normalizeVolume, nullptr);
        trackTree.setProperty("nsfSmoothPitchTransitions", track.nsfSmoothPitchTransitions, nullptr);
        trackTree.setProperty("nsfNoiseProtection", track.nsfNoiseProtection, nullptr);
        trackTree.setProperty("nativeNsfAudio", track.nativeNsfAudio, nullptr);
        trackTree.setProperty("voicebankDirectory",
                              track.voicebankDirectory.getFullPathName(), nullptr);
        trackTree.setProperty("utauConsonantVelocity", track.utauConsonantVelocity, nullptr);
        trackTree.setProperty("diffSingerLanguage", track.diffSingerLanguage, nullptr);
        trackTree.setProperty("diffSingerSpeaker", track.diffSingerSpeaker, nullptr);
        trackTree.setProperty("diffSingerDictionary", track.diffSingerDictionary, nullptr);
        trackTree.setProperty("utauGlobalFlags", track.utauGlobalFlags, nullptr);
        trackTree.setProperty("ustSourceDocument",track.ustSourceDocument,nullptr);
        trackTree.setProperty("ustSourceBytes",track.ustSourceBytes,nullptr);
        trackTree.setProperty("ustSourceEncoding",track.ustSourceEncoding,nullptr);
        trackTree.setProperty("ustSourceBom",track.ustSourceBom,nullptr);
        trackTree.setProperty("ustBaseline",track.ustBaseline,nullptr);
        trackTree.setProperty("utauMode", utauModeKey(track.utauMode), nullptr);
        trackTree.setProperty("utauPhonemizer", track.chineseCvvc ? "zh-cvvc" : "manual", nullptr);
        trackTree.setProperty("utauPianoPreview", track.utauPianoPreview, nullptr);
        trackTree.setProperty("cvvcCrossfade",track.cvvcCurveCrossfade?"curve":"linear",nullptr);
        trackTree.setProperty("outputEngine", utauOutputEngineKey(track.outputEngine), nullptr);
        trackTree.setProperty("outputResampler", track.outputResampler.getFullPathName(), nullptr);
        trackTree.setProperty("outputWavtool", track.outputWavtool.getFullPathName(), nullptr);
        if (projectFile != juce::File{})
            for (const auto& entry : { std::make_pair("outputResampler", track.outputResampler),
                                     std::make_pair("outputWavtool", track.outputWavtool) })
                if (entry.second != juce::File{})
                    trackTree.setProperty(juce::String(entry.first) + "Relative",
                        entry.second.getRelativePathFrom(projectFile.getParentDirectory()), nullptr);
        // Still written so a build from before 谋 opens the project in the
        // nearest mode it has rather than dropping to plain UTAU.
        trackTree.setProperty("utauFourRegion",
                              utauModeUsesRegions(track.utauMode), nullptr);
        if (projectFile != juce::File{} && track.voicebankDirectory != juce::File{})
        {
            const auto relative = track.voicebankDirectory.getRelativePathFrom(
                projectFile.getParentDirectory());
            if (relative.isNotEmpty() && !juce::File::isAbsolutePath(relative))
                trackTree.setProperty("voicebankDirectoryRelative", relative, nullptr);
        }
        trackTree.setProperty("pitchAlgorithm", pitchAlgorithmName(track.pitchAlgorithm), nullptr);
        trackTree.setProperty("trackType",trackTypeName(effectiveTrackType(track)),nullptr);
        trackTree.setProperty("stretchAlgorithm", stretchAlgorithmName(track.stretchAlgorithm), nullptr);
        trackTree.setProperty("renderOrder", renderOrderName(track.renderOrder), nullptr);
        trackTree.setProperty("mergeContiguousSource", track.mergeContiguousSource, nullptr);

        for (const auto& clip : track.clips)
        {
            juce::ValueTree clipTree("Clip");
            clipTree.setProperty("id", clip.id, nullptr);
            clipTree.setProperty("sourceFile", clip.sourceFile.getFullPathName(), nullptr);
            clipTree.setProperty("materialSourceFingerprint",clip.materialSourceFingerprint,nullptr);
            if(clip.materialSourceUsageGroupId.isNotEmpty())
                clipTree.setProperty("materialSourceUsageGroupId",clip.materialSourceUsageGroupId,nullptr);
            clipTree.setProperty("materialSourceAnalysisPending",clip.materialSourceAnalysisPending,nullptr);
            clipTree.setProperty("materialSourceResumeMuted",clip.materialSourceResumeMuted,nullptr);
            if (projectFile != juce::File{} && clip.sourceFile != juce::File{})
            {
                const auto relative = clip.sourceFile.getRelativePathFrom(
                    projectFile.getParentDirectory());
                if (relative.isNotEmpty() && !juce::File::isAbsolutePath(relative))
                    clipTree.setProperty("sourceFileRelative", relative, nullptr);
            }
            clipTree.setProperty("startSeconds", clip.startSeconds, nullptr);
            clipTree.setProperty("sourceOffsetSeconds", clip.sourceOffsetSeconds, nullptr);
            clipTree.setProperty("sourceDurationSeconds", clip.sourceDurationSeconds, nullptr);
            clipTree.setProperty("durationSeconds", clip.durationSeconds, nullptr);
            clipTree.setProperty("audioStartSeconds", clip.audioStartSeconds, nullptr);
            clipTree.setProperty("audioDurationSeconds", clip.audioDurationSeconds, nullptr);
            clipTree.setProperty("fadeInSeconds", clip.fadeInSeconds, nullptr);
            clipTree.setProperty("fadeOutSeconds", clip.fadeOutSeconds, nullptr);
            clipTree.setProperty("crossfadeInSeconds", clip.crossfadeInSeconds, nullptr);
            clipTree.setProperty("crossfadeOutSeconds", clip.crossfadeOutSeconds, nullptr);
            clipTree.setProperty("gain", clip.gain, nullptr);
            writeClipGainEnvelope(clipTree,clip);
            clipTree.setProperty("muted", clip.muted, nullptr);
            clipTree.setProperty("showNoteHints", clip.showNoteHints, nullptr);
            clipTree.setProperty("showNormalDisplay", clip.showNormalDisplay, nullptr);
            clipTree.setProperty("nativeAudioLinked", clip.nativeAudioLinked, nullptr);
            writeNativeTrimReference(clipTree,clip);
            clipTree.setProperty("glideConnectedToNext", clip.glideConnectedToNext, nullptr);
            clipTree.setProperty("glideConnectedFromPrevious", clip.glideConnectedFromPrevious, nullptr);

            for (const auto& point : clip.sourceTimeMap)
            {
                juce::ValueTree pointTree("SourceTimePoint");
                pointTree.setProperty("targetSeconds", point.targetSeconds, nullptr);
                pointTree.setProperty("sourceSeconds", point.sourceSeconds, nullptr);
                clipTree.addChild(pointTree, -1, nullptr);
            }

            for (const auto& note : clip.notes)
            {
                juce::ValueTree noteTree("Note");
                noteTree.setProperty("id", note.id, nullptr);
                if(note.clipPartId.isNotEmpty())noteTree.setProperty("clipPartId",note.clipPartId,nullptr);
                noteTree.setProperty("label", note.label, nullptr);
                noteTree.setProperty("materialAudioId", note.materialAudioId, nullptr);
                noteTree.setProperty("materialRegionId", note.materialRegionId, nullptr);
                noteTree.setProperty("materialBoundAlias",note.materialBoundAlias,nullptr);
                noteTree.setProperty("materialAliasFollowsPrimary",note.materialAliasFollowsPrimary,nullptr);
                noteTree.setProperty("materialLabelFollowsAlias",note.materialLabelFollowsAlias,nullptr);
                noteTree.setProperty("materialTargetPitchLocked",note.materialTargetPitchLocked,nullptr);
                if(note.pitchSupportAuthored)noteTree.setProperty("pitchSupportAuthored",true,nullptr);
                noteTree.setProperty("materialSourceStartSeconds", note.materialSourceStartSeconds, nullptr);
                noteTree.setProperty("materialSourceEndSeconds", note.materialSourceEndSeconds, nullptr);
                noteTree.setProperty("materialLocalOverride", note.materialLocalOverride, nullptr);
                noteTree.setProperty("materialReferenceUnresolved", note.materialReferenceUnresolved, nullptr);
                noteTree.setProperty("materialAdditionalAliases", juce::var(note.materialAdditionalAliases), nullptr);
                noteTree.setProperty("materialConnectedNextRegionId", note.materialConnectedNextRegionId, nullptr);
                noteTree.setProperty("pronunciationKind",pronunciationKindName(note.pronunciationKind),nullptr);
                noteTree.setProperty("simplifiedConsonant",note.simplifiedConsonant,nullptr);
                noteTree.setProperty("vowelSplitsExplicit",note.vowelSplitsExplicit,nullptr);
                noteTree.setProperty("nativeRole", nativeSegmentRoleName(note.nativeRole), nullptr);
                noteTree.setProperty("nativeProvenance", note.nativeProvenance, nullptr);
                noteTree.setProperty("nativeConfidence", note.nativeConfidence, nullptr);
                noteTree.setProperty("nativeSourceStartSeconds",
                    note.nativeSourceStartSeconds, nullptr);
                noteTree.setProperty("nativeSourceEndSeconds",
                    note.nativeSourceEndSeconds, nullptr);
                noteTree.setProperty("utauFlags", note.utauFlags, nullptr);
                noteTree.setProperty("diffSingerTiming", note.diffSingerTiming, nullptr);
                noteTree.setProperty("diffSingerPronunciation", note.diffSingerPronunciation, nullptr);
                noteTree.setProperty("vibratoEnabled", note.vibratoEnabled, nullptr);
                noteTree.setProperty("vibratoLengthPercent", note.vibratoLengthPercent, nullptr);
                noteTree.setProperty("vibratoCycleMs", note.vibratoCycleMs, nullptr);
                noteTree.setProperty("vibratoDepthCents", note.vibratoDepthCents, nullptr);
                noteTree.setProperty("vibratoFadeInPercent", note.vibratoFadeInPercent, nullptr);
                noteTree.setProperty("vibratoFadeOutPercent", note.vibratoFadeOutPercent, nullptr);
                noteTree.setProperty("vibratoPhasePercent", note.vibratoPhasePercent, nullptr);
                noteTree.setProperty("vibratoOffsetPercent", note.vibratoOffsetPercent, nullptr);
                noteTree.setProperty("vibratoEndPercent", note.vibratoEndPercent, nullptr);
                noteTree.setProperty("vibratoReferenceDurationSeconds", note.vibratoReferenceDurationSeconds, nullptr);
                noteTree.setProperty("vibratoTimeOffsetSeconds", note.vibratoTimeOffsetSeconds, nullptr);
                noteTree.setProperty("vibratoRealLine", note.vibratoRealLine, nullptr);
                noteTree.setProperty("utauTailFadeMode",note.utauTailFadeMode,nullptr);
                noteTree.setProperty("utauTailFadeStartFraction",note.utauTailFade.startFraction,nullptr);
                noteTree.setProperty("utauTailFadeEndFraction",note.utauTailFade.endFraction,nullptr);
                noteTree.setProperty("utauTailFadeStartGain",note.utauTailFade.startGain,nullptr);
                noteTree.setProperty("utauTailFadeEndGain",note.utauTailFade.endGain,nullptr);
                noteTree.setProperty("utauTailFadeCurvePower",note.utauTailFade.curvePower,nullptr);
                noteTree.setProperty("utauHeadEnvelopeMode",note.utauTailFade.head.mode,nullptr);
                noteTree.setProperty("utauHeadEnvelopeStartFraction",note.utauTailFade.head.startFraction,nullptr);
                noteTree.setProperty("utauHeadEnvelopeEndFraction",note.utauTailFade.head.endFraction,nullptr);
                noteTree.setProperty("utauHeadEnvelopeStartGain",note.utauTailFade.head.startGain,nullptr);
                noteTree.setProperty("utauHeadEnvelopeEndGain",note.utauTailFade.head.endGain,nullptr);
                noteTree.setProperty("utauHeadEnvelopeCurvePower",note.utauTailFade.head.curvePower,nullptr);
                noteTree.setProperty("utauHeadEnvelopeCustomCurve",note.utauTailFade.head.customCurve,nullptr);
                noteTree.setProperty("utauHeadEnvelopeControl1Time",note.utauTailFade.head.control1Time,nullptr);
                noteTree.setProperty("utauHeadEnvelopeControl1Progress",note.utauTailFade.head.control1Progress,nullptr);
                noteTree.setProperty("utauHeadEnvelopeControl2Time",note.utauTailFade.head.control2Time,nullptr);
                noteTree.setProperty("utauHeadEnvelopeControl2Progress",note.utauTailFade.head.control2Progress,nullptr);
                noteTree.setProperty("utauTailFadeCustomCurve",note.utauTailFade.customCurve,nullptr);
                noteTree.setProperty("utauTailFadeControl1Time",note.utauTailFade.control1Time,nullptr);
                noteTree.setProperty("utauTailFadeControl1Progress",note.utauTailFade.control1Progress,nullptr);
                noteTree.setProperty("utauTailFadeControl2Time",note.utauTailFade.control2Time,nullptr);
                noteTree.setProperty("utauTailFadeControl2Progress",note.utauTailFade.control2Progress,nullptr);
                noteTree.setProperty("utauTailMixed",backend::mixedEnvelopeText(note.utauTailFade),nullptr);
                noteTree.setProperty("utauHeadMixed",backend::mixedEnvelopeText(note.utauTailFade.head),nullptr);
                noteTree.setProperty("nativeEnvelope",juce::JSON::toString(backend::nativeEnvelopeToVar(note.nativeEnvelope),true,17),nullptr);
                noteTree.setProperty("amplitudeEnvelopeBasePercent",
                                     note.amplitudeEnvelopeBasePercent, nullptr);
                noteTree.setProperty("utauFlagSplit", note.utauFlagSplit, nullptr);
                noteTree.setProperty("utauFlagCurveEnabled",
                                     note.utauFlagCurveEnabled, nullptr);
                if (trackIsDiffSinger(track))
                    noteTree.setProperty("diffSingerOffsetEnabled", note.utauFlagCurveEnabled, nullptr);
                noteTree.setProperty("utauSplice", note.utauSplice, nullptr);
                noteTree.setProperty("utauAutoPitchTransition",
                                     note.utauAutoPitchTransition, nullptr);
                noteTree.setProperty("nativeIndependentPitch", note.nativeIndependentPitch, nullptr);
                noteTree.setProperty("nativeUnpitched", note.nativeUnpitched, nullptr);
                noteTree.setProperty("nativePitchHandlesPlaced", note.nativePitchHandlesPlaced, nullptr);
                noteTree.setProperty("utauRegionFlags1", note.utauRegionFlags1, nullptr);
                noteTree.setProperty("utauRegionFlags2", note.utauRegionFlags2, nullptr);
                noteTree.setProperty("utauRegionFlags3", note.utauRegionFlags3, nullptr);
                noteTree.setProperty("utauRegionFlags4", note.utauRegionFlags4, nullptr);
                noteTree.setProperty("utauJieSplitSet", note.utauJieSplitSet, nullptr);
                noteTree.setProperty("utauJieSplit1", note.utauJieSplit1, nullptr);
                noteTree.setProperty("utauJieSplit2", note.utauJieSplit2, nullptr);
                noteTree.setProperty("utauJieSplit3", note.utauJieSplit3, nullptr);
                noteTree.setProperty("utauConsonantVelocity",
                                     note.utauConsonantVelocity, nullptr);
                noteTree.setProperty("utauConsonantVelocityInherited",
                    note.utauConsonantVelocity == inheritedUtauConsonantVelocity,
                    nullptr);
                noteTree.setProperty("utauPreutteranceOverrideEnabled",
                    note.utauPreutteranceOverrideEnabled, nullptr);
                noteTree.setProperty("utauPreutteranceSeconds",
                    note.utauPreutteranceSeconds, nullptr);
                noteTree.setProperty("utauOverlapOverrideEnabled",
                    note.utauOverlapOverrideEnabled, nullptr);
                noteTree.setProperty("utauOverlapSeconds",
                    note.utauOverlapSeconds, nullptr);
                noteTree.setProperty("utauStpSeconds", note.utauStpSeconds, nullptr);
                noteTree.setProperty("utauModulationPercent",note.utauModulationPercent,nullptr);
                noteTree.setProperty("ustSourceSection",note.ustSourceSection,nullptr);
                noteTree.setProperty("ustSourceSectionIndex",note.ustSourceSectionIndex,nullptr);
                noteTree.setProperty("ustBaseline",note.ustBaseline,nullptr);
                if (note.utauOto.enabled)
                {
                    noteTree.setProperty("utauOto", true, nullptr);
                    noteTree.setProperty("utauOtoOffsetMs", note.utauOto.offsetMs, nullptr);
                    noteTree.setProperty("utauOtoConsonantMs", note.utauOto.consonantMs, nullptr);
                    noteTree.setProperty("utauOtoCutoffMs", note.utauOto.cutoffMs, nullptr);
                    noteTree.setProperty("utauOtoPreutteranceMs", note.utauOto.preutteranceMs, nullptr);
                    noteTree.setProperty("utauOtoOverlapMs", note.utauOto.overlapMs, nullptr);
                    noteTree.setProperty("utauOtoOnsetMs", note.utauOto.onsetMs, nullptr);
                    noteTree.setProperty("utauOtoGlideMs", note.utauOto.glideMs, nullptr);
                    noteTree.setProperty("utauOtoNucleusMs", note.utauOto.nucleusMs, nullptr);
                    noteTree.setProperty("utauOtoHasRegions", note.utauOto.hasRegions, nullptr);
                    noteTree.setProperty("utauOtoClasses", note.utauOto.classes, nullptr);
                }
                noteTree.setProperty("startSeconds", note.startSeconds, nullptr);
                noteTree.setProperty("durationSeconds", note.durationSeconds, nullptr);
                noteTree.setProperty("consonantSeconds", note.consonantSeconds, nullptr);
                noteTree.setProperty("melodyneConsonantCandidate",
                    note.melodyneConsonantCandidate, nullptr);
                noteTree.setProperty("melodyneVowelNoteId",
                    note.melodyneVowelNoteId, nullptr);
                noteTree.setProperty("midiNote", note.midiNote, nullptr);
                noteTree.setProperty("sourceMidiCenter", note.sourceMidiCenter, nullptr);
                noteTree.setProperty("sourcePitchMeasured", note.sourcePitchMeasured, nullptr);
                noteTree.setProperty("modulation", note.modulation, nullptr);
                noteTree.setProperty("drift", note.drift, nullptr);
                noteTree.setProperty("tension", note.tension, nullptr);
                noteTree.setProperty("breath", note.breath, nullptr);
                noteTree.setProperty("formantSemitones", note.formantSemitones, nullptr);
                noteTree.setProperty("gain", note.gain, nullptr);
                noteTree.setProperty("attackSpeed", note.attackSpeed, nullptr);
                noteTree.setProperty("utauTailFadeMode",note.utauTailFadeMode,nullptr);
                noteTree.setProperty("utauTailFadeStartFraction",note.utauTailFade.startFraction,nullptr);
                noteTree.setProperty("utauTailFadeEndFraction",note.utauTailFade.endFraction,nullptr);
                noteTree.setProperty("utauTailFadeStartGain",note.utauTailFade.startGain,nullptr);
                noteTree.setProperty("utauTailFadeEndGain",note.utauTailFade.endGain,nullptr);
                noteTree.setProperty("utauTailFadeCurvePower",note.utauTailFade.curvePower,nullptr);
                noteTree.setProperty("utauHeadEnvelopeMode",note.utauTailFade.head.mode,nullptr);
                noteTree.setProperty("utauHeadEnvelopeStartFraction",note.utauTailFade.head.startFraction,nullptr);
                noteTree.setProperty("utauHeadEnvelopeEndFraction",note.utauTailFade.head.endFraction,nullptr);
                noteTree.setProperty("utauHeadEnvelopeStartGain",note.utauTailFade.head.startGain,nullptr);
                noteTree.setProperty("utauHeadEnvelopeEndGain",note.utauTailFade.head.endGain,nullptr);
                noteTree.setProperty("utauHeadEnvelopeCurvePower",note.utauTailFade.head.curvePower,nullptr);
                noteTree.setProperty("utauHeadEnvelopeCustomCurve",note.utauTailFade.head.customCurve,nullptr);
                noteTree.setProperty("utauHeadEnvelopeControl1Time",note.utauTailFade.head.control1Time,nullptr);
                noteTree.setProperty("utauHeadEnvelopeControl1Progress",note.utauTailFade.head.control1Progress,nullptr);
                noteTree.setProperty("utauHeadEnvelopeControl2Time",note.utauTailFade.head.control2Time,nullptr);
                noteTree.setProperty("utauHeadEnvelopeControl2Progress",note.utauTailFade.head.control2Progress,nullptr);
                noteTree.setProperty("utauTailFadeCustomCurve",note.utauTailFade.customCurve,nullptr);
                noteTree.setProperty("utauTailFadeControl1Time",note.utauTailFade.control1Time,nullptr);
                noteTree.setProperty("utauTailFadeControl1Progress",note.utauTailFade.control1Progress,nullptr);
                noteTree.setProperty("utauTailFadeControl2Time",note.utauTailFade.control2Time,nullptr);
                noteTree.setProperty("utauTailFadeControl2Progress",note.utauTailFade.control2Progress,nullptr);
                noteTree.setProperty("utauTailMixed",backend::mixedEnvelopeText(note.utauTailFade),nullptr);
                noteTree.setProperty("utauHeadMixed",backend::mixedEnvelopeText(note.utauTailFade.head),nullptr);
                noteTree.setProperty("nativeEnvelope",juce::JSON::toString(backend::nativeEnvelopeToVar(note.nativeEnvelope),true,17),nullptr);
                noteTree.setProperty("amplitudeEnvelopeBase",
                                     note.amplitudeEnvelopeBasePercent, nullptr);
                if(!note.cvvcBoundaries.empty())noteTree.setProperty("cvvcBoundaries",
                    juce::JSON::toString(backend::cvvcBoundariesToVar(note.cvvcBoundaries),true,17),nullptr);
                if(!note.cvvcEnvelopes.empty())noteTree.setProperty("cvvcEnvelopes",
                    juce::JSON::toString(backend::cvvcEnvelopesToVar(note.cvvcEnvelopes),true,17),nullptr);
                if(includeDerived&&note.cvvcPhonemeCache.isNotEmpty())
                    noteTree.setProperty("cvvcPhonemeCache",note.cvvcPhonemeCache,nullptr);
                noteTree.setProperty("robustPitchCurve", note.robustPitchCurve, nullptr);
                noteTree.setProperty("connectedToPrevious", note.connectedToPrevious, nullptr);
                noteTree.setProperty("connectedToNext", note.connectedToNext, nullptr);
                for (const auto& point : note.contour)
                {
                    juce::ValueTree pointTree("PitchPoint");
                    pointTree.setProperty("timeSeconds", point.timeSeconds, nullptr);
                    pointTree.setProperty("relativeCents", point.relativeCents, nullptr);
                    pointTree.setProperty("withoutVibratoCents", point.withoutVibratoCents, nullptr);
                    pointTree.setProperty("voiced", point.voiced, nullptr);
                    pointTree.setProperty("manualTargetCents", point.manualTargetCents, nullptr);
                    pointTree.setProperty("hasManualTarget", point.hasManualTarget, nullptr);
                    noteTree.addChild(pointTree, -1, nullptr);
                }
                for (const auto& point : note.pitchControlPoints)
                {
                    juce::ValueTree pointTree("PitchControlPoint");
                    if(point.highPrecisionBezier)pointTree.setProperty("highPrecisionBezier",true,nullptr);
                    pointTree.setProperty("timeSeconds", point.timeSeconds, nullptr);
                    pointTree.setProperty("targetMidi", point.targetMidi, nullptr);
                    pointTree.setProperty("shape", pitchCurveShapeName(point.shape), nullptr);
                    pointTree.setProperty("bezierX1", point.bezierX1, nullptr);
                    pointTree.setProperty("bezierY1", point.bezierY1, nullptr);
                    pointTree.setProperty("bezierX2", point.bezierX2, nullptr);
                    pointTree.setProperty("bezierY2", point.bezierY2, nullptr);
                    if (point.diffSingerRestoreSupport)
                        pointTree.setProperty("diffSingerRestoreSupport", true, nullptr);
                    noteTree.addChild(pointTree, -1, nullptr);
                }
                if (!note.diffSingerPitchReference.empty() && note.diffSingerPitchReferenceFromSavedPitch)
                    noteTree.setProperty("diffSingerPitchReferenceFromSavedPitch", true, nullptr);
                for (const auto& point : note.diffSingerPitchOffset)
                {
                    juce::ValueTree pointTree("DiffSingerPitchOffsetPoint");
                    pointTree.setProperty("timeSeconds",point.timeSeconds,nullptr);
                    pointTree.setProperty("semitones",point.targetMidi,nullptr);
                    noteTree.addChild(pointTree,-1,nullptr);
                }
                for (const auto& point : note.diffSingerPitchReference)
                {
                    juce::ValueTree pointTree("DiffSingerPitchReferencePoint");
                    if(point.highPrecisionBezier)pointTree.setProperty("highPrecisionBezier",true,nullptr);
                    pointTree.setProperty("timeSeconds", point.timeSeconds, nullptr);
                    pointTree.setProperty("targetMidi", point.targetMidi, nullptr);
                    pointTree.setProperty("shape", pitchCurveShapeName(point.shape), nullptr);
                    pointTree.setProperty("bezierX1", point.bezierX1, nullptr);
                    pointTree.setProperty("bezierY1", point.bezierY1, nullptr);
                    pointTree.setProperty("bezierX2", point.bezierX2, nullptr);
                    pointTree.setProperty("bezierY2", point.bezierY2, nullptr);
                    noteTree.addChild(pointTree, -1, nullptr);
                }
                for (const auto& point : note.amplitudeEnvelope)
                {
                    juce::ValueTree pointTree("AmplitudeEnvelopePoint");
                    pointTree.setProperty("timeSeconds", point.timeSeconds, nullptr);
                    pointTree.setProperty("gainDb", point.gainDb, nullptr);
                    if (point.linearToNext)
                        pointTree.setProperty("linearToNext", true, nullptr);
                    if (point.nativeSeamAnchor) pointTree.setProperty("nativeSeamAnchor", true, nullptr);
                    noteTree.addChild(pointTree, -1, nullptr);
                }
                for (const auto& segment : note.nativeSegments)
                {
                    juce::ValueTree segmentTree("NativeSegment");
                    segmentTree.setProperty("id", segment.id, nullptr);
                    segmentTree.setProperty("alias", segment.alias, nullptr);
                    segmentTree.setProperty("role", nativeSegmentRoleName(segment.role), nullptr);
                    segmentTree.setProperty("sourceStartSeconds", segment.sourceStartSeconds, nullptr);
                    segmentTree.setProperty("sourceEndSeconds", segment.sourceEndSeconds, nullptr);
                    segmentTree.setProperty("provenance", segment.provenance, nullptr);
                    segmentTree.setProperty("confidence", segment.confidence, nullptr);
                    segmentTree.setProperty("alignmentSeconds", segment.alignmentSeconds, nullptr);
                    segmentTree.setProperty("overlapSeconds", segment.overlapSeconds, nullptr);
                    segmentTree.setProperty("stretchable", segment.stretchable, nullptr);
                    segmentTree.setProperty("stretchWeight", segment.stretchWeight, nullptr);
                    noteTree.addChild(segmentTree, -1, nullptr);
                }
                for (const auto& curve : note.utauFlagCurves)
                {
                    if (curve.points.empty() && curve.flag.startsWith("HIFI:")) {
                        juce::ValueTree overrideTree("FlagCurveOverride");
                        overrideTree.setProperty("flag",curve.flag,nullptr);
                        noteTree.addChild(overrideTree,-1,nullptr);
                    }
                for (const auto& point : curve.points)
                {
                    juce::ValueTree pointTree("FlagCurvePoint");
                    pointTree.setProperty("flag", curve.flag, nullptr);
                    pointTree.setProperty("timeSeconds", point.timeSeconds, nullptr);
                    pointTree.setProperty("value", point.value, nullptr);
                    pointTree.setProperty("shape", pitchCurveShapeName(point.shape),
                                          nullptr);
                    pointTree.setProperty("bezierX1", point.bezierX1, nullptr);
                    pointTree.setProperty("bezierY1", point.bezierY1, nullptr);
                    pointTree.setProperty("bezierX2", point.bezierX2, nullptr);
                    pointTree.setProperty("bezierY2", point.bezierY2, nullptr);
                    noteTree.addChild(pointTree, -1, nullptr);
                }
                }
                for (const auto marker : note.sibilantMarkers)
                {
                    juce::ValueTree markerTree("Sibilant");
                    markerTree.setProperty("timeSeconds", marker, nullptr);
                    noteTree.addChild(markerTree, -1, nullptr);
                }
                clipTree.addChild(noteTree, -1, nullptr);
            }
            if(!clip.parts.empty())
            {
                ProjectModel partsModel;
                partsModel.project.tracks.clear();
                TrackData sources;sources.id="sources";sources.clips=clip.parts;
                partsModel.project.tracks.push_back(std::move(sources));
                juce::ValueTree partsTree("ClipParts");
                partsTree.addChild(partsModel.toValueTree(projectFile,includeDerived),-1,nullptr);
                clipTree.addChild(partsTree,-1,nullptr);
            }
            trackTree.addChild(clipTree, -1, nullptr);
        }
        root.addChild(trackTree, -1, nullptr);
    }
    return root;
}

ProjectData ProjectModel::fromValueTree(const juce::ValueTree& root,
                                        const juce::File& projectFile)
{
    ProjectData data;
    data.hamoodState=root["hamoodState"].toString();
    const auto projectDirectory = projectFile.getParentDirectory();
    std::map<juce::String, juce::File> recursiveMedia;
    auto indexedMedia = false;
    const auto resolveSource = [&](const juce::ValueTree& clipTree)
    {
        const auto storedPath = clipTree.getProperty("sourceFile").toString();
        juce::File source(storedPath);
        if (source.existsAsFile()) return source;
        const auto relative = clipTree.getProperty("sourceFileRelative").toString();
        if (relative.isNotEmpty())
        {
            const auto candidate = projectDirectory.getChildFile(relative);
            if (candidate.existsAsFile()) return candidate;
        }
        const auto fileName = source.getFileName();
        if (fileName.isNotEmpty())
        {
            const auto besideProject = projectDirectory.getChildFile(fileName);
            if (besideProject.existsAsFile()) return besideProject;
            if (!indexedMedia && projectDirectory.isDirectory())
            {
                indexedMedia = true;
                juce::Array<juce::File> files;
                projectDirectory.findChildFiles(files, juce::File::findFiles, true);
                for (const auto& file : files)
                    recursiveMedia.try_emplace(file.getFileName().toLowerCase(), file);
            }
            if (const auto found = recursiveMedia.find(fileName.toLowerCase());
                found != recursiveMedia.end()) return found->second;
        }
        return source;
    };
    data.name = root.getProperty("name", "Untitled").toString();
    data.bpm = static_cast<double>(root.getProperty("bpm", 120.0));
    data.beatOriginSeconds = static_cast<double>(root.getProperty("beatOriginSeconds", 0.0));
    data.numerator = static_cast<int>(root.getProperty("numerator", 4));
    data.denominator = static_cast<int>(root.getProperty("denominator", 4));
    data.gridDivision = root.getProperty("gridDivision", "1/16").toString();
    data.noteEditDivision = juce::jlimit(2, 128,
        static_cast<int>(root.getProperty("noteEditDivision", 64)));
    data.baseScale = root.getProperty("baseScale", "C").toString();
    data.assetManagerVisible = static_cast<bool>(root.getProperty("assetManagerVisible", false));

    for (const auto child : root)
        if (child.hasType("TempoChange"))
        {
            const auto position = std::max(0.0,
                static_cast<double>(child.getProperty("quarterPosition", 0.0)));
            const auto tempo = juce::jlimit(20.0, 400.0,
                static_cast<double>(child.getProperty("bpm", data.bpm)));
            if (position > 1.0e-7) data.tempoChanges.push_back({ position, tempo });
        }
    std::stable_sort(data.tempoChanges.begin(), data.tempoChanges.end(),
        [](const auto& left, const auto& right)
        {
            return left.quarterPosition < right.quarterPosition;
        });

    for (const auto connectionTree : root)
        if (connectionTree.hasType("NativeConnection"))
        {
            NativeConnection connection;
            connection.id = connectionTree.getProperty("id").toString();
            connection.leftNoteId = connectionTree.getProperty("leftNoteId").toString();
            connection.rightNoteId = connectionTree.getProperty("rightNoteId").toString();
            connection.type = connectionTree.getProperty("type", "pitch-and-amplitude").toString();
            connection.boundarySeconds = static_cast<double>(
                connectionTree.getProperty("boundarySeconds", 0.0));
            connection.pitchCurveEndpointRelative=static_cast<bool>(connectionTree.getProperty("pitchCurveEndpointRelative",false));
            connection.pitchStartOffsetSeconds=static_cast<double>(connectionTree.getProperty("pitchStartOffsetSeconds",-.05));
            connection.pitchEndOffsetSeconds=static_cast<double>(connectionTree.getProperty("pitchEndOffsetSeconds",.05));
            for (const auto pointTree : connectionTree)
            {
                if (pointTree.hasType("PitchCurvePoint"))
                    connection.pitchCurve.push_back({
                        static_cast<double>(pointTree.getProperty("timeSeconds", 0.0)),
                        static_cast<float>(pointTree.getProperty("targetMidi", 60.0)),
                        parsePitchCurveShape(pointTree.getProperty("shape", "natural").toString()),
                        static_cast<float>(pointTree.getProperty("bezierX1", 0.33)),
                        static_cast<float>(pointTree.getProperty("bezierY1", 0.0)),
                        static_cast<float>(pointTree.getProperty("bezierX2", 0.67)),
                        static_cast<float>(pointTree.getProperty("bezierY2", 1.0)),false,
                        static_cast<bool>(pointTree.getProperty("highPrecisionBezier",false)) });
                else if (pointTree.hasType("AmplitudeCurvePoint"))
                    connection.amplitudeCurve.push_back({
                        static_cast<double>(pointTree.getProperty("timeSeconds", 0.0)),
                        static_cast<float>(pointTree.getProperty("gainDb", 0.0)) });
            }
            data.nativeConnections.push_back(std::move(connection));
        }

    for (const auto trackTree : root)
    {
        if (!trackTree.hasType("Track")) continue;
        TrackData track;
        if(trackTree.hasProperty("trackType"))
        {
            // The load entry validates explicit types before any conversion,
            // including nested source-part trees. Missing remains legacy input.
            track.trackType=parseTrackType(trackTree.getProperty("trackType").toString());
        }
        track.id = trackTree.getProperty("id").toString();
        track.name = trackTree.getProperty("name").toString();
        track.accompaniment = static_cast<bool>(trackTree.getProperty("accompaniment", false));
        track.compose = !track.accompaniment && static_cast<bool>(trackTree.getProperty("compose", true));
        track.muted = static_cast<bool>(trackTree.getProperty("muted", false));
        track.solo = static_cast<bool>(trackTree.getProperty("solo", false));
        track.referenceOnly =
            !track.accompaniment && static_cast<bool>(trackTree.getProperty("referenceOnly", false));
        track.volume = static_cast<float>(trackTree.getProperty("volume", 1.0));
        std::vector<GainEnvelopePoint> legacyTrackEnvelope;
        for (const auto pointTree : trackTree)
            if (pointTree.hasType("TrackGainPoint"))
                legacyTrackEnvelope.push_back({static_cast<double>(pointTree.getProperty("timeSeconds", 0.0)),
                    static_cast<float>(pointTree.getProperty("gainDb", 0.0))});
        legacyTrackEnvelope = normaliseTrackGainEnvelope(std::move(legacyTrackEnvelope));
        track.pan = static_cast<float>(trackTree.getProperty("pan", 0.0));
        track.smoothOverlaps = static_cast<bool>(trackTree.getProperty("smoothOverlaps", false));
        track.allowNativeAudioOverlap=static_cast<bool>(trackTree.getProperty("allowNativeAudioOverlap",false));
        track.normalizeVolume = static_cast<bool>(trackTree.getProperty("normalizeVolume", false));
        track.nsfSmoothPitchTransitions = static_cast<bool>(trackTree.getProperty("nsfSmoothPitchTransitions", true));
        track.nsfNoiseProtection = static_cast<bool>(trackTree.getProperty("nsfNoiseProtection", false));
        track.nativeNsfAudio = static_cast<bool>(trackTree.getProperty("nativeNsfAudio", false));
        track.voicebankDirectory = juce::File(
            trackTree.getProperty("voicebankDirectory").toString());
        track.utauConsonantVelocity = static_cast<int>(
            trackTree.getProperty("utauConsonantVelocity", 100));
        track.utauGlobalFlags = trackTree.getProperty("utauGlobalFlags").toString();
        track.ustSourceDocument=trackTree.getProperty("ustSourceDocument").toString();
        track.ustSourceBytes=trackTree.getProperty("ustSourceBytes").toString();
        track.ustSourceEncoding=trackTree.getProperty("ustSourceEncoding").toString();
        track.ustSourceBom=(bool)trackTree.getProperty("ustSourceBom",false);
        track.ustBaseline=trackTree.getProperty("ustBaseline").toString();
        track.diffSingerLanguage = trackTree.getProperty("diffSingerLanguage", "zh").toString();
        track.diffSingerSpeaker = trackTree.getProperty("diffSingerSpeaker").toString();
        track.diffSingerDictionary = trackTree.getProperty("diffSingerDictionary").toString();
        track.chineseCvvc = trackTree.getProperty("utauPhonemizer", "manual").toString() == "zh-cvvc";
        track.utauPianoPreview = static_cast<bool>(trackTree.getProperty("utauPianoPreview", true));
        track.cvvcCurveCrossfade = trackTree.getProperty("cvvcCrossfade", "linear").toString()=="curve";
        // A project written before 谋 carries only the flag.
        track.utauMode = trackTree.hasProperty("utauMode")
            ? parseUtauMode(trackTree.getProperty("utauMode", "").toString())
            : (static_cast<bool>(trackTree.getProperty("utauFourRegion", false))
                   ? UtauMode::jie : UtauMode::classic);
        if (!track.voicebankDirectory.isDirectory())
        {
            const auto relative = trackTree.getProperty("voicebankDirectoryRelative").toString();
            if (relative.isNotEmpty())
            {
                const auto candidate = projectDirectory.getChildFile(relative);
                if (candidate.isDirectory()) track.voicebankDirectory = candidate;
            }
        }
        track.pitchAlgorithm = parsePitchAlgorithm(trackTree.getProperty("pitchAlgorithm", "mld5").toString());
        track.outputEngine = parseUtauOutputEngine(trackTree.getProperty("outputEngine").toString());
        const auto engineFile = [&](const char* key) {
            juce::File file(trackTree.getProperty(key).toString());
            const auto relative = trackTree.getProperty(juce::String(key) + "Relative").toString();
            if (!file.existsAsFile() && relative.isNotEmpty()) {
                const auto candidate = projectDirectory.getChildFile(relative);
                if (candidate.existsAsFile()) file = candidate;
            }
            return file;
        };
        track.outputResampler = engineFile("outputResampler");
        track.outputWavtool = engineFile("outputWavtool");
        if (!track.trackType && !track.nativeNsfAudio && !trackTree.hasProperty("outputEngine") && track.pitchAlgorithm == PitchAlgorithm::nsfHifigan
            && !track.accompaniment && (utauModeUsesRegions(track.utauMode) || track.voicebankDirectory != juce::File{})
            && !track.voicebankDirectory.getChildFile("dsconfig.yaml").existsAsFile()) {
            track.pitchAlgorithm = PitchAlgorithm::utau;
            track.outputEngine = UtauOutputEngine::pcNsfHifigan;
        }
        if (!track.trackType&&track.voicebankDirectory.getChildFile("dsconfig.yaml").existsAsFile()) track.utauMode = UtauMode::mou;
        if(!track.trackType)track.trackType=effectiveTrackType(track);
        track.stretchAlgorithm = parseStretchAlgorithm(trackTree.getProperty("stretchAlgorithm", "melodyne-hybrid").toString());
        track.renderOrder = parseRenderOrder(trackTree.getProperty("renderOrder", "process-then-splice").toString());
        track.mergeContiguousSource = static_cast<bool>(trackTree.getProperty("mergeContiguousSource", false));

        for (const auto clipTree : trackTree)
        {
            if (!clipTree.hasType("Clip")) continue;
            ClipData clip;
            clip.id = clipTree.getProperty("id").toString();
            clip.materialSourceFingerprint=clipTree.getProperty("materialSourceFingerprint").toString();
            clip.materialSourceUsageGroupId=clipTree.getProperty("materialSourceUsageGroupId").toString();
            clip.materialSourceAnalysisPending=static_cast<bool>(clipTree.getProperty("materialSourceAnalysisPending",false));
            clip.materialSourceResumeMuted=static_cast<bool>(clipTree.getProperty("materialSourceResumeMuted",false));
            clip.sourceFile = resolveSource(clipTree);
            clip.startSeconds = static_cast<double>(clipTree.getProperty("startSeconds", 0.0));
            clip.sourceOffsetSeconds = static_cast<double>(clipTree.getProperty("sourceOffsetSeconds", 0.0));
            clip.sourceDurationSeconds = static_cast<double>(clipTree.getProperty("sourceDurationSeconds", 0.0));
            clip.durationSeconds = static_cast<double>(clipTree.getProperty("durationSeconds", 1.0));
            // The editable boundary may be shorter than the retained audio window.
            // Preserve that window so expanding the region after reopening is lossless.
            const auto audioStart = static_cast<double>(clipTree.getProperty("audioStartSeconds", 0.0));
            clip.audioStartSeconds = std::isfinite(audioStart) ? std::max(0.0, audioStart) : 0.0;
            clip.audioDurationSeconds = static_cast<double>(clipTree.getProperty("audioDurationSeconds", -1.0));
            if (!std::isfinite(clip.audioDurationSeconds)) clip.audioDurationSeconds = -1.0;
            clip.fadeInSeconds = static_cast<double>(clipTree.getProperty("fadeInSeconds", 0.0));
            clip.fadeOutSeconds = static_cast<double>(clipTree.getProperty("fadeOutSeconds", 0.0));
            clip.crossfadeInSeconds = static_cast<double>(clipTree.getProperty("crossfadeInSeconds", 0.0));
            clip.crossfadeOutSeconds = static_cast<double>(clipTree.getProperty("crossfadeOutSeconds", 0.0));
            clip.gain = static_cast<float>(clipTree.getProperty("gain", 1.0));
            readClipGainEnvelope(clipTree,clip);
            if(!legacyTrackEnvelope.empty())
            {
                auto local=legacyTrackEnvelope;shiftGainEnvelope(local,-clip.startSeconds);
                if(clip.gainEnvelope.empty()){clip.gainEnvelope=std::move(local);anchorClipGainEnvelope(clip);}
                else clip.inheritedGainEnvelopes.push_back(std::move(local));
            }
            clip.muted = static_cast<bool>(clipTree.getProperty("muted", false));
            clip.showNoteHints = static_cast<bool>(clipTree.getProperty("showNoteHints", false));
            clip.showNormalDisplay = static_cast<bool>(clipTree.getProperty("showNormalDisplay", false));
            clip.nativeAudioLinked = static_cast<bool>(clipTree.getProperty("nativeAudioLinked", false));
            readNativeTrimReference(clipTree,clip);
            clip.glideConnectedToNext = static_cast<bool>(
                clipTree.getProperty("glideConnectedToNext", false));
            clip.glideConnectedFromPrevious = static_cast<bool>(
                clipTree.getProperty("glideConnectedFromPrevious", false));

            for (const auto noteTree : clipTree)
            {
                if(noteTree.hasType("ClipParts"))
                {
                    auto sources=fromValueTree(noteTree.getChild(0),projectFile);
                    if(!sources.tracks.empty())clip.parts=std::move(sources.tracks.front().clips);
                    continue;
                }
                if (noteTree.hasType("SourceTimePoint"))
                {
                    clip.sourceTimeMap.push_back({
                        static_cast<double>(noteTree.getProperty("targetSeconds", 0.0)),
                        static_cast<double>(noteTree.getProperty("sourceSeconds", 0.0)) });
                    continue;
                }
                if (!noteTree.hasType("Note")) continue;
                NoteData note;
                note.id = noteTree.getProperty("id").toString();
                note.clipPartId = noteTree.getProperty("clipPartId").toString();
                note.label = noteTree.getProperty("label").toString();
                note.materialAudioId = noteTree.getProperty("materialAudioId").toString();
                note.materialRegionId = noteTree.getProperty("materialRegionId").toString();
                note.materialBoundAlias=noteTree.getProperty("materialBoundAlias").toString();
                note.materialAliasFollowsPrimary=static_cast<bool>(noteTree.getProperty("materialAliasFollowsPrimary",true));
                // Old files cannot distinguish renamed project labels from a
                // source alias. Preserve their display conservatively until an
                // explicit source/alias choice supplies the missing intent.
                note.materialLabelFollowsAlias=static_cast<bool>(noteTree.getProperty("materialLabelFollowsAlias",note.materialAudioId.isEmpty()));
                note.materialTargetPitchLocked=static_cast<bool>(noteTree.getProperty("materialTargetPitchLocked",false));
                note.pitchSupportAuthored=static_cast<bool>(noteTree.getProperty("pitchSupportAuthored",false));
                note.materialSourceStartSeconds = static_cast<double>(noteTree.getProperty("materialSourceStartSeconds", -1.0));
                note.materialSourceEndSeconds = static_cast<double>(noteTree.getProperty("materialSourceEndSeconds", -1.0));
                note.materialLocalOverride = static_cast<bool>(noteTree.getProperty("materialLocalOverride", false));
                note.materialReferenceUnresolved = static_cast<bool>(noteTree.getProperty("materialReferenceUnresolved", false));
                if (const auto* aliases = noteTree.getProperty("materialAdditionalAliases").getArray())
                    for (const auto& alias : *aliases) note.materialAdditionalAliases.add(alias.toString());
                note.materialConnectedNextRegionId = noteTree.getProperty("materialConnectedNextRegionId").toString();
                if(noteTree.hasProperty("pronunciationKind"))
                    note.pronunciationKind=parsePronunciationKind(noteTree.getProperty("pronunciationKind").toString());
                else if(static_cast<bool>(noteTree.getProperty("materialTransitionPronunciation",false)))
                    note.pronunciationKind=PronunciationKind::transition;
                else
                {
                    // Legacy role is INTERNAL metadata and is parsed separately
                    // below. Only old breath/transition identity is recovered;
                    // old consonant/vowel elements remain structurally untouched.
                    const auto legacyRole=parseNativeSegmentRole(noteTree.getProperty("nativeRole","unknown").toString());
                    note.pronunciationKind=legacyRole==NativeSegmentRole::breath?PronunciationKind::breath
                        :legacyRole==NativeSegmentRole::transition?PronunciationKind::transition:PronunciationKind::normal;
                }
                note.simplifiedConsonant=static_cast<bool>(noteTree.getProperty("simplifiedConsonant",false));
                note.vowelSplitsExplicit=static_cast<bool>(noteTree.getProperty("vowelSplitsExplicit",false));
                note.nativeRole = parseNativeSegmentRole(
                    noteTree.getProperty("nativeRole", "unknown").toString());
                note.nativeProvenance = noteTree.getProperty("nativeProvenance", "estimated").toString();
                note.nativeConfidence = static_cast<float>(
                    noteTree.getProperty("nativeConfidence", 0.0));
                note.nativeSourceStartSeconds = static_cast<double>(
                    noteTree.getProperty("nativeSourceStartSeconds", -1.0));
                note.nativeSourceEndSeconds = static_cast<double>(
                    noteTree.getProperty("nativeSourceEndSeconds", -1.0));
                note.utauFlags = noteTree.getProperty("utauFlags").toString();
                note.diffSingerTiming = noteTree.getProperty("diffSingerTiming").toString();
                note.diffSingerPronunciation = noteTree.getProperty("diffSingerPronunciation").toString();
                note.diffSingerPitchReferenceFromSavedPitch = static_cast<bool>(
                    noteTree.getProperty("diffSingerPitchReferenceFromSavedPitch", false));
                const auto storedVelocity = static_cast<int>(
                    noteTree.getProperty("utauConsonantVelocity", -1));
                note.utauJieSplitSet = static_cast<bool>(
                    noteTree.getProperty("utauJieSplitSet", false));
                note.vibratoEnabled = static_cast<bool>(
                    noteTree.getProperty("vibratoEnabled", false));
                note.vibratoLengthPercent = noteTree.getProperty("vibratoLengthPercent", 65.0);
                note.vibratoCycleMs = noteTree.getProperty("vibratoCycleMs", 180.0);
                note.vibratoDepthCents = noteTree.getProperty("vibratoDepthCents", 35.0);
                note.vibratoFadeInPercent = noteTree.getProperty("vibratoFadeInPercent", 20.0);
                note.vibratoFadeOutPercent = noteTree.getProperty("vibratoFadeOutPercent", 20.0);
                note.vibratoPhasePercent = noteTree.getProperty("vibratoPhasePercent", 0.0);
                note.vibratoOffsetPercent = noteTree.getProperty("vibratoOffsetPercent", 0.0);
                // Older projects have none, and 100 is where UTAU stops.
                note.vibratoEndPercent = juce::jlimit(1.0, 100.0, static_cast<double>(
                    noteTree.getProperty("vibratoEndPercent", 100.0)));
                note.vibratoReferenceDurationSeconds = std::max(0.0, static_cast<double>(noteTree.getProperty("vibratoReferenceDurationSeconds", 0.0)));
                note.vibratoTimeOffsetSeconds = std::max(0.0, static_cast<double>(noteTree.getProperty("vibratoTimeOffsetSeconds", 0.0)));
                note.vibratoRealLine = static_cast<bool>(
                    noteTree.getProperty("vibratoRealLine", false));
                note.amplitudeEnvelopeBasePercent = juce::jlimit(0.0f, 200.0f,
                    static_cast<float>(static_cast<double>(
                        noteTree.getProperty("amplitudeEnvelopeBasePercent", 100.0))));
                note.cvvcBoundaries=backend::cvvcBoundariesFromVar(juce::JSON::parse(noteTree.getProperty("cvvcBoundaries").toString()));
                note.cvvcEnvelopes=backend::cvvcEnvelopesFromVar(juce::JSON::parse(noteTree.getProperty("cvvcEnvelopes").toString()));
                const auto phonemes=noteTree.getProperty("cvvcPhonemeCache").toString();
                if(!backend::readCvvcPhonemeCache(phonemes).spans.empty())note.cvvcPhonemeCache=phonemes;
                note.utauTailFadeMode=juce::jlimit(0,2,(int)noteTree.getProperty("utauTailFadeMode",0));
                note.nativeEnvelope=backend::nativeEnvelopeFromVar(juce::JSON::parse(noteTree.getProperty("nativeEnvelope").toString()));
                note.utauTailFade.startFraction=(double)noteTree.getProperty("utauTailFadeStartFraction",0.0);
                note.utauTailFade.endFraction=(double)noteTree.getProperty("utauTailFadeEndFraction",1.0);
                note.utauTailFade.startGain=(double)noteTree.getProperty("utauTailFadeStartGain",1.0);
                note.utauTailFade.endGain=(double)noteTree.getProperty("utauTailFadeEndGain",0.0);
                note.utauTailFade.curvePower=(double)noteTree.getProperty("utauTailFadeCurvePower",1.0);
                note.utauTailFade.customCurve=(bool)noteTree.getProperty("utauTailFadeCustomCurve",false);
                note.utauTailFade.control1Time=(double)noteTree.getProperty("utauTailFadeControl1Time",1.0/3.0);
                note.utauTailFade.control1Progress=(double)noteTree.getProperty("utauTailFadeControl1Progress",0.0);
                note.utauTailFade.control2Time=(double)noteTree.getProperty("utauTailFadeControl2Time",2.0/3.0);
                note.utauTailFade.control2Progress=(double)noteTree.getProperty("utauTailFadeControl2Progress",1.0);
                note.utauTailFade.head.mode=(int)noteTree.getProperty("utauHeadEnvelopeMode",0);
                note.utauTailFade.head.startFraction=(double)noteTree.getProperty("utauHeadEnvelopeStartFraction",0.0);
                note.utauTailFade.head.endFraction=(double)noteTree.getProperty("utauHeadEnvelopeEndFraction",1.0);
                note.utauTailFade.head.startGain=(double)noteTree.getProperty("utauHeadEnvelopeStartGain",0.0);
                note.utauTailFade.head.endGain=(double)noteTree.getProperty("utauHeadEnvelopeEndGain",1.0);
                note.utauTailFade.head.curvePower=(double)noteTree.getProperty("utauHeadEnvelopeCurvePower",1.0);
                note.utauTailFade.head.customCurve=(bool)noteTree.getProperty("utauHeadEnvelopeCustomCurve",false);
                note.utauTailFade.head.control1Time=(double)noteTree.getProperty("utauHeadEnvelopeControl1Time",1.0/3.0);
                note.utauTailFade.head.control1Progress=(double)noteTree.getProperty("utauHeadEnvelopeControl1Progress",0.0);
                note.utauTailFade.head.control2Time=(double)noteTree.getProperty("utauHeadEnvelopeControl2Time",2.0/3.0);
                note.utauTailFade.head.control2Progress=(double)noteTree.getProperty("utauHeadEnvelopeControl2Progress",1.0);
                if(!backend::mixedEnvelopeFromVar(juce::JSON::parse(noteTree.getProperty("utauTailMixed").toString()),note.utauTailFade)
                    ||!backend::mixedEnvelopeFromVar(juce::JSON::parse(noteTree.getProperty("utauHeadMixed").toString()),note.utauTailFade.head)
                    ||!note.utauTailFade.valid())note.utauTailFade={};
                note.utauFlagSplit = static_cast<bool>(
                    noteTree.getProperty("utauFlagSplit", false));
                // Distinguish a deliberate disable from the old default-off flag.
                note.utauFlagCurveEnabled = trackIsDiffSinger(track)
                    ? static_cast<bool>(noteTree.getProperty("diffSingerOffsetEnabled", true))
                    : static_cast<bool>(noteTree.getProperty("utauFlagCurveEnabled", false));
                note.utauSplice = static_cast<bool>(
                    noteTree.getProperty("utauSplice", false));
                // A project saved before this was read carries no such
                // property, and its notes glided; they still do.
                note.utauAutoPitchTransition = static_cast<bool>(
                    noteTree.getProperty("utauAutoPitchTransition", true));
                note.nativeIndependentPitch = static_cast<bool>(noteTree.getProperty("nativeIndependentPitch", false));
                note.nativeUnpitched = static_cast<bool>(noteTree.getProperty("nativeUnpitched", false));
                note.nativePitchHandlesPlaced = static_cast<bool>(noteTree.getProperty("nativePitchHandlesPlaced", false));
                note.utauRegionFlags1 = noteTree.getProperty("utauRegionFlags1").toString();
                note.utauRegionFlags2 = noteTree.getProperty("utauRegionFlags2").toString();
                note.utauRegionFlags3 = noteTree.getProperty("utauRegionFlags3").toString();
                note.utauRegionFlags4 = noteTree.getProperty("utauRegionFlags4").toString();
                note.utauJieSplit1 = noteTree.getProperty("utauJieSplit1", 0.0);
                note.utauJieSplit2 = noteTree.getProperty("utauJieSplit2", 0.0);
                note.utauJieSplit3 = noteTree.getProperty("utauJieSplit3", 0.0);
                if (noteTree.hasProperty("utauConsonantVelocityInherited"))
                    note.utauConsonantVelocity = static_cast<bool>(noteTree.getProperty(
                        "utauConsonantVelocityInherited", false))
                            ? inheritedUtauConsonantVelocity : storedVelocity;
                else
                    // Before signed velocity support, every negative value was
                    // normalised to -1 and meant "use the track value".
                    note.utauConsonantVelocity = storedVelocity < 0
                        ? inheritedUtauConsonantVelocity : storedVelocity;
                note.utauPreutteranceOverrideEnabled = static_cast<bool>(
                    noteTree.getProperty("utauPreutteranceOverrideEnabled", false));
                note.utauPreutteranceSeconds = std::max(0.0, static_cast<double>(
                    noteTree.getProperty("utauPreutteranceSeconds", 0.0)));
                note.utauOverlapOverrideEnabled = static_cast<bool>(
                    noteTree.getProperty("utauOverlapOverrideEnabled", false));
                note.utauOverlapSeconds = static_cast<double>(
                    noteTree.getProperty("utauOverlapSeconds", 0.0));
                note.utauStpSeconds = static_cast<double>(
                    noteTree.getProperty("utauStpSeconds", 0.0));
                note.utauModulationPercent=(double)noteTree.getProperty("utauModulationPercent",0.0);
                note.ustSourceSection=noteTree.getProperty("ustSourceSection").toString();
                note.ustSourceSectionIndex=(int)noteTree.getProperty("ustSourceSectionIndex",-1);
                note.ustBaseline=noteTree.getProperty("ustBaseline").toString();
                note.utauOto.enabled = static_cast<bool>(
                    noteTree.getProperty("utauOto", false));
                if (note.utauOto.enabled)
                {
                    note.utauOto.offsetMs = static_cast<double>(
                        noteTree.getProperty("utauOtoOffsetMs", 0.0));
                    note.utauOto.consonantMs = static_cast<double>(
                        noteTree.getProperty("utauOtoConsonantMs", 0.0));
                    note.utauOto.cutoffMs = static_cast<double>(
                        noteTree.getProperty("utauOtoCutoffMs", 0.0));
                    note.utauOto.preutteranceMs = static_cast<double>(
                        noteTree.getProperty("utauOtoPreutteranceMs", 0.0));
                    note.utauOto.overlapMs = static_cast<double>(
                        noteTree.getProperty("utauOtoOverlapMs", 0.0));
                    note.utauOto.onsetMs = static_cast<double>(
                        noteTree.getProperty("utauOtoOnsetMs", 0.0));
                    note.utauOto.glideMs = static_cast<double>(
                        noteTree.getProperty("utauOtoGlideMs", 0.0));
                    note.utauOto.nucleusMs = static_cast<double>(
                        noteTree.getProperty("utauOtoNucleusMs", 0.0));
                    note.utauOto.hasRegions = static_cast<bool>(
                        noteTree.getProperty("utauOtoHasRegions", false));
                    note.utauOto.classes = noteTree.getProperty("utauOtoClasses").toString();
                }
                note.startSeconds = static_cast<double>(noteTree.getProperty("startSeconds", 0.0));
                note.durationSeconds = static_cast<double>(noteTree.getProperty("durationSeconds", 0.25));
                note.consonantSeconds = static_cast<double>(noteTree.getProperty("consonantSeconds", 0.04));
                note.melodyneConsonantCandidate = static_cast<bool>(
                    noteTree.getProperty("melodyneConsonantCandidate", false));
                note.melodyneVowelNoteId = noteTree.getProperty(
                    "melodyneVowelNoteId").toString();
                note.midiNote = static_cast<float>(noteTree.getProperty("midiNote", 60.0));
                note.sourceMidiCenter = static_cast<float>(noteTree.getProperty("sourceMidiCenter", -1.0));
                note.sourcePitchMeasured = static_cast<bool>(noteTree.getProperty("sourcePitchMeasured", false));
                note.modulation = static_cast<float>(noteTree.getProperty("modulation", 1.0));
                note.drift = static_cast<float>(noteTree.getProperty("drift", 1.0));
                note.tension = static_cast<float>(noteTree.getProperty("tension", 0.0));
                note.breath = static_cast<float>(noteTree.getProperty("breath", 0.0));
                note.formantSemitones = static_cast<float>(noteTree.getProperty("formantSemitones", 0.0));
                note.gain = static_cast<float>(noteTree.getProperty("gain", 1.0));
                note.attackSpeed = static_cast<float>(noteTree.getProperty("attackSpeed", 1.0));
                // Older projects have no base, and 100 is the envelope as drawn.
                note.amplitudeEnvelopeBasePercent = juce::jlimit(0.0f, 200.0f,
                    static_cast<float>(noteTree.getProperty("amplitudeEnvelopeBase", 100.0)));
                note.utauTailFadeMode=juce::jlimit(0,2,(int)noteTree.getProperty("utauTailFadeMode",0));
                note.nativeEnvelope=backend::nativeEnvelopeFromVar(juce::JSON::parse(noteTree.getProperty("nativeEnvelope").toString()));
                note.utauTailFade.startFraction=(double)noteTree.getProperty("utauTailFadeStartFraction",0.0);
                note.utauTailFade.endFraction=(double)noteTree.getProperty("utauTailFadeEndFraction",1.0);
                note.utauTailFade.startGain=(double)noteTree.getProperty("utauTailFadeStartGain",1.0);
                note.utauTailFade.endGain=(double)noteTree.getProperty("utauTailFadeEndGain",0.0);
                note.utauTailFade.curvePower=(double)noteTree.getProperty("utauTailFadeCurvePower",1.0);
                note.utauTailFade.customCurve=(bool)noteTree.getProperty("utauTailFadeCustomCurve",false);
                note.utauTailFade.control1Time=(double)noteTree.getProperty("utauTailFadeControl1Time",1.0/3.0);
                note.utauTailFade.control1Progress=(double)noteTree.getProperty("utauTailFadeControl1Progress",0.0);
                note.utauTailFade.control2Time=(double)noteTree.getProperty("utauTailFadeControl2Time",2.0/3.0);
                note.utauTailFade.control2Progress=(double)noteTree.getProperty("utauTailFadeControl2Progress",1.0);
                note.utauTailFade.head.mode=(int)noteTree.getProperty("utauHeadEnvelopeMode",0);
                note.utauTailFade.head.startFraction=(double)noteTree.getProperty("utauHeadEnvelopeStartFraction",0.0);
                note.utauTailFade.head.endFraction=(double)noteTree.getProperty("utauHeadEnvelopeEndFraction",1.0);
                note.utauTailFade.head.startGain=(double)noteTree.getProperty("utauHeadEnvelopeStartGain",0.0);
                note.utauTailFade.head.endGain=(double)noteTree.getProperty("utauHeadEnvelopeEndGain",1.0);
                note.utauTailFade.head.curvePower=(double)noteTree.getProperty("utauHeadEnvelopeCurvePower",1.0);
                note.utauTailFade.head.customCurve=(bool)noteTree.getProperty("utauHeadEnvelopeCustomCurve",false);
                note.utauTailFade.head.control1Time=(double)noteTree.getProperty("utauHeadEnvelopeControl1Time",1.0/3.0);
                note.utauTailFade.head.control1Progress=(double)noteTree.getProperty("utauHeadEnvelopeControl1Progress",0.0);
                note.utauTailFade.head.control2Time=(double)noteTree.getProperty("utauHeadEnvelopeControl2Time",2.0/3.0);
                note.utauTailFade.head.control2Progress=(double)noteTree.getProperty("utauHeadEnvelopeControl2Progress",1.0);
                if(!backend::mixedEnvelopeFromVar(juce::JSON::parse(noteTree.getProperty("utauTailMixed").toString()),note.utauTailFade)
                    ||!backend::mixedEnvelopeFromVar(juce::JSON::parse(noteTree.getProperty("utauHeadMixed").toString()),note.utauTailFade.head)
                    ||!note.utauTailFade.valid())note.utauTailFade={};
                note.robustPitchCurve = static_cast<bool>(
                    noteTree.getProperty("robustPitchCurve", false));
                note.connectedToPrevious = static_cast<bool>(noteTree.getProperty("connectedToPrevious", false));
                note.connectedToNext = static_cast<bool>(noteTree.getProperty("connectedToNext", false));
                for (const auto child : noteTree)
                {
                    if (child.hasType("PitchPoint"))
                    {
                        const auto relative = static_cast<float>(child.getProperty("relativeCents", 0.0));
                        note.contour.push_back({ static_cast<double>(child.getProperty("timeSeconds", 0.0)),
                                                 relative,
                                                 static_cast<float>(child.getProperty("withoutVibratoCents", relative)),
                                                 static_cast<bool>(child.getProperty("voiced", true)),
                                                 static_cast<float>(child.getProperty("manualTargetCents", 0.0)),
                                                 static_cast<bool>(child.getProperty("hasManualTarget", false)) });
                    }
                    else if (child.hasType("PitchControlPoint"))
                        note.pitchControlPoints.push_back({
                            static_cast<double>(child.getProperty("timeSeconds", 0.0)),
                            static_cast<float>(child.getProperty("targetMidi", note.midiNote)),
                            parsePitchCurveShape(child.getProperty("shape", "natural").toString()),
                            static_cast<float>(child.getProperty("bezierX1", 0.33)),
                            static_cast<float>(child.getProperty("bezierY1", 0.0)),
                            static_cast<float>(child.getProperty("bezierX2", 0.67)),
                            static_cast<float>(child.getProperty("bezierY2", 1.0)),
                            static_cast<bool>(child.getProperty("diffSingerRestoreSupport", false)),
                            static_cast<bool>(child.getProperty("highPrecisionBezier",false)) });
                    else if (child.hasType("DiffSingerPitchOffsetPoint"))
                    {
                        const auto time=static_cast<double>(child.getProperty("timeSeconds",0.0));
                        const auto value=static_cast<float>(child.getProperty("semitones",0.0));
                        if (std::isfinite(time) && std::isfinite(value))
                            note.diffSingerPitchOffset.push_back({juce::jlimit(0.0,note.durationSeconds,time),
                                juce::jlimit(-127.0f,127.0f,value),PitchCurveShape::linear});
                    }
                    else if (child.hasType("DiffSingerPitchReferencePoint"))
                        note.diffSingerPitchReference.push_back({
                            static_cast<double>(child.getProperty("timeSeconds", 0.0)),
                            static_cast<float>(child.getProperty("targetMidi", note.midiNote)),
                            parsePitchCurveShape(child.getProperty("shape", "natural").toString()),
                            static_cast<float>(child.getProperty("bezierX1", 0.33)),
                            static_cast<float>(child.getProperty("bezierY1", 0.0)),
                            static_cast<float>(child.getProperty("bezierX2", 0.67)),
                            static_cast<float>(child.getProperty("bezierY2", 1.0)),false,
                            static_cast<bool>(child.getProperty("highPrecisionBezier",false)) });
                    else if (child.hasType("AmplitudeEnvelopePoint"))
                        note.amplitudeEnvelope.push_back({
                            static_cast<double>(child.getProperty("timeSeconds", 0.0)),
                            juce::jlimit(-60.0f, 12.0f,
                                static_cast<float>(child.getProperty("gainDb", 0.0))),
                            static_cast<bool>(child.getProperty("linearToNext", false)),
                            static_cast<bool>(child.getProperty("nativeSeamAnchor", false)) });
                    else if (child.hasType("NativeSegment"))
                        note.nativeSegments.push_back({
                            child.getProperty("id", "segment").toString(),
                            child.getProperty("alias", "-").toString(),
                            parseNativeSegmentRole(child.getProperty("role", "unknown").toString()),
                            static_cast<double>(child.getProperty("sourceStartSeconds", 0.0)),
                            static_cast<double>(child.getProperty("sourceEndSeconds", 0.0)),
                            child.getProperty("provenance", "estimated").toString(),
                            static_cast<float>(child.getProperty("confidence", 0.0)),
                            static_cast<double>(child.getProperty("alignmentSeconds", 0.0)),
                            static_cast<double>(child.getProperty("overlapSeconds", 0.0)),
                            static_cast<bool>(child.getProperty("stretchable", true)),
                            static_cast<double>(child.getProperty("stretchWeight", 1.0)) });
                    // "FlagCurveG" is how the g curve was written before any
                    // other flag could have one; it reads as the g curve.
                    else if (child.hasType("FlagCurveOverride"))
                    {
                        const auto flag=child.getProperty("flag").toString();
                        if (flag.startsWith("HIFI:") && std::none_of(note.utauFlagCurves.begin(),note.utauFlagCurves.end(),
                            [&](const auto& c){return c.flag==flag;})) note.utauFlagCurves.push_back({flag,{}});
                    }
                    else if (child.hasType("FlagCurvePoint")
                             || child.hasType("FlagCurveG"))
                    {
                        const auto flag = child.hasType("FlagCurveG")
                            ? juce::String("g")
                            : child.getProperty("flag", "g").toString();
                        const auto& kind = flagCurveKindFor(flag);
                        auto found = std::find_if(note.utauFlagCurves.begin(),
                            note.utauFlagCurves.end(),
                            [&flag](const auto& curve) { return curve.flag == flag; });
                        if (found == note.utauFlagCurves.end())
                        {
                            note.utauFlagCurves.push_back({ flag, {} });
                            found = std::prev(note.utauFlagCurves.end());
                        }
                        found->points.push_back({
                            static_cast<double>(child.getProperty("timeSeconds", 0.0)),
                            juce::jlimit(kind.minimum, kind.maximum,
                                static_cast<float>(child.getProperty("value", 0.0))),
                            parsePitchCurveShape(
                                child.getProperty("shape", "linear").toString()),
                            static_cast<float>(child.getProperty("bezierX1", 0.33)),
                            static_cast<float>(child.getProperty("bezierY1", 0.0)),
                            static_cast<float>(child.getProperty("bezierX2", 0.67)),
                            static_cast<float>(child.getProperty("bezierY2", 1.0)) });
                    }
                    else if (child.hasType("Sibilant"))
                        note.sibilantMarkers.push_back(static_cast<double>(child.getProperty("timeSeconds", 0.0)));
                }
                for (auto& curve : note.utauFlagCurves)
                    std::stable_sort(curve.points.begin(), curve.points.end(),
                        [](const auto& left, const auto& right)
                        {
                            return left.timeSeconds < right.timeSeconds;
                        });
                // Updates before 088 stored measured source curves as dense
                // PitchPoints, without a provenance bit. Two-point authored
                // notes and voicebank/MIDI tracks must not acquire that bit.
                if (!noteTree.hasProperty("sourcePitchMeasured")
                    && !trackUsesVoicebankSynthesis(track) && !trackIsDiffSinger(track)
                    && note.nativeProvenance != "utau" && note.sourceMidiCenter >= 0.0f
                    && note.contour.size() > 2)
                    note.sourcePitchMeasured = true;
                std::stable_sort(note.amplitudeEnvelope.begin(),
                                 note.amplitudeEnvelope.end(),
                    [](const auto& left, const auto& right)
                    {
                        return left.timeSeconds < right.timeSeconds;
                    });
                // Old CVVC files materialised the default whole-note envelope
                // without an interpolation flag. Those dB ramps survived the
                // independent-phoneme defaults and multiplied the mixer again
                // at the phrase head/tail. Migrate only the exact old 5/15 ms,
                // 35 ms unity template; authored shapes and modern files keep
                // their semantics. No OTO scan or file write is needed here.
                if (!root.hasProperty("cvvcEnvelopeVersion") && track.chineseCvvc
                    && trackUsesVoicebankSynthesis(track) && !trackIsDiffSinger(track)
                    && note.utauTailFadeMode==0 && note.utauTailFade.head.mode==0
                    && note.ustSourceSection.isEmpty() && note.amplitudeEnvelope.size()==4)
                {
                    const auto& p=note.amplitudeEnvelope;
                    const auto near=[](double a,double b){return std::abs(a-b)<1.e-7;};
                    const auto attack=p[1].timeSeconds-p[0].timeSeconds;
                    if (std::all_of(p.begin(),p.end(),[](const auto& v){return !v.linearToNext&&!v.nativeSeamAnchor;})
                        && near(p[0].gainDb,-60)&&near(p[1].gainDb,0)&&near(p[2].gainDb,0)&&near(p[3].gainDb,-60)
                        && p[0].timeSeconds<=0 && (near(attack,.005)||near(attack,.015))
                        && near(p[3].timeSeconds-p[2].timeSeconds,.035)
                        && near(p[3].timeSeconds,note.durationSeconds) && p[1].timeSeconds<p[2].timeSeconds)
                        note.amplitudeEnvelope.clear();
                }
                // Before the reference field existed, DS files saved only the
                // current pitch. Keep that as an explicitly labelled load-time
                // reference; never replace a genuine saved prediction or create
                // a reference for a new MIDI-only note in a current-format file.
                std::stable_sort(note.diffSingerPitchOffset.begin(),note.diffSingerPitchOffset.end(),
                    [](const auto& a,const auto& b){return a.timeSeconds<b.timeSeconds;});
                if (!root.hasProperty("diffSingerPitchReferenceVersion")
                    && trackIsDiffSinger(track) && note.diffSingerPitchReference.empty()
                    && (!note.pitchControlPoints.empty()
                        || std::any_of(note.contour.begin(), note.contour.end(),
                            [](const auto& p) { return p.hasManualTarget; })))
                {
                    note.diffSingerPitchReference = ownPitchPoints(note);
                    note.diffSingerPitchReferenceFromSavedPitch = !note.diffSingerPitchReference.empty();
                }
                clip.notes.push_back(std::move(note));
            }
            std::stable_sort(clip.sourceTimeMap.begin(), clip.sourceTimeMap.end(),
                [](const auto& left, const auto& right)
                {
                    return left.targetSeconds < right.targetSeconds;
                });
            auto previousTarget = -1.0;
            auto previousSource = -1.0;
            std::erase_if(clip.sourceTimeMap, [&](auto& point)
            {
                point.targetSeconds = juce::jlimit(0.0, clip.durationSeconds,
                                                   point.targetSeconds);
                point.sourceSeconds = juce::jlimit(0.0,
                    clip.sourceDurationSeconds > 0.0 ? clip.sourceDurationSeconds
                                                     : clip.durationSeconds,
                    point.sourceSeconds);
                const auto invalid = !std::isfinite(point.targetSeconds)
                    || !std::isfinite(point.sourceSeconds)
                    || point.targetSeconds <= previousTarget + 1.0e-9
                    || point.sourceSeconds < previousSource - 1.0e-9;
                if (!invalid)
                {
                    previousTarget = point.targetSeconds;
                    previousSource = point.sourceSeconds;
                }
                return invalid;
            });
            if (clip.sourceTimeMap.size() < 2) clip.sourceTimeMap.clear();
            track.clips.push_back(std::move(clip));
        }
        data.tracks.push_back(std::move(track));
    }
    return data;
}

bool ProjectModel::save(const juce::File& file, juce::String& error) const
{
    juce::MemoryOutputStream bytes;
    toValueTree(file).writeToStream(bytes);
    return projectio::save(file, bytes.getMemoryBlock(), error, true);
}

bool ProjectModel::saveRecoverySnapshot(ProjectData data, const juce::File& recoveryFile,
                                         const juce::File& originalFile, juce::String& error)
{
    ProjectModel captured;
    captured.project = std::move(data);
    // Paths are relative to the recovery file; absolute paths are also retained.
    auto tree = captured.toValueTree(recoveryFile);
    tree.setProperty("recoveryOriginalFile", originalFile.getFullPathName(), nullptr);
    tree.setProperty("recoverySavedAt", juce::Time::getCurrentTime().toISO8601(true), nullptr);
    juce::MemoryOutputStream bytes;
    tree.writeToStream(bytes);
    return projectio::save(recoveryFile, bytes.getMemoryBlock(), error, false);
}

bool ProjectModel::load(const juce::File& file, juce::String& error)
{
    error.clear();
    // Read the file whole, then parse it out of memory.  ValueTree reads a
    // stream a field at a time and a FileInputStream turns every one of those
    // into its own read of the disk, which is why opening a project took
    // seconds and got worse the bigger it was: 970 KB was 2.8 s off the file
    // against 30 ms from a block the same file loaded into in under a
    // millisecond.  Writing never had this -- 26 ms for the same project --
    // so only this side needed it.
    juce::MemoryBlock bytes;
    if (file.loadFileAsData(bytes))
    {
        auto tree = projectio::validatedTree(bytes);
        if (tree.hasType("HachiShifterProject"))
        {
            std::function<bool(const juce::ValueTree&)> validTrackTypes;
            validTrackTypes=[&](const juce::ValueTree& node)
            {
                if(node.hasType("Track")&&node.hasProperty("trackType")&&!parseTrackType(node.getProperty("trackType").toString()))
                {error="Unsupported explicit track type: "+node.getProperty("trackType").toString();return false;}
                for(const auto child:node)if(!validTrackTypes(child))return false;
                return true;
            };
            if(!validTrackTypes(tree))return false;
            auto loaded = fromValueTree(tree, file);
            juce::StringArray missing;
            for (const auto& track : loaded.tracks)
                for (const auto& clip : track.clips)
                    if (clip.sourceFile != juce::File{} && !clip.sourceFile.existsAsFile())
                        missing.addIfNotAlreadyThere(clip.sourceFile.getFullPathName());
            resetDocument(std::move(loaded));
            if (!missing.isEmpty())
                error = missing.joinIntoString("\n");
            return true;
        }
    }
    error = "Invalid HachiShifter Next project: " + file.getFullPathName();
    return false;
}
}

#include "ProjectMaterialRegistrationModel.inc"
