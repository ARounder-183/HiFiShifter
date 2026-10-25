#pragma once

// Adapted from OpenUtau's ChineseCVVCPhonemizer (MIT).
// See third_party/openutau/NOTICE.md and LICENSE.txt.
#include "UtauRenderer.h"
#include "LegacyTextCodec.h"
#include "../Pinyin.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <optional>

namespace hachi::backend
{
struct ChineseCvvcRules
{
    std::map<juce::String, juce::String> vowels, vowelNames, consonants, replacements;
    // Defaults retain the ZH CVVC behaviour for banks without ALIAS rules.
    std::map<juce::String, juce::String> aliasFormats {
        {"VCV", "%v%%VCVPAD%%CV%"}, {"BEGINING_CV", "-%VCVPAD%%CV%"},
        {"CV", "%CV%"}, {"VC", "%v%%VCPAD%%c%"}, {"ENDING1", "%v%%VCPAD%R"}
    };
    juce::String vcPad = " ", vcvPad = " ";
    juce::StringArray priorities;

    // Only equivalent Mandarin umlaut spellings, not a replacement phoneme
    // dictionary. A bank's exact rules and explicit REPLACE always win.
    static juce::StringArray spellingVariants(const juce::String& syllable)
    {
        for (const auto* group : { "nve,nue,nüe,nu:e", "lve,lue,lüe,lu:e",
                                  "nv,nü,nu:", "lv,lü,lu:" })
        {
            auto variants = juce::StringArray::fromTokens(juce::String::fromUTF8(group), ",", "");
            if (variants.contains(syllable)) return variants;
        }
        return {};
    }

    static ChineseCvvcRules parse(const juce::String& text)
    {
        ChineseCvvcRules rules;
        juce::String section;
        std::optional<juce::String> legacyPad, legacyEnd, legacyEnd1;
        for (auto raw : juce::StringArray::fromLines(text))
        {
            raw = raw.trimCharactersAtStart(juce::String::charToString(0xfeff));
            auto line = raw.trim();
            if (line.startsWithChar(';') || line.startsWithChar('#')) continue;
            if (line.startsWithChar('[')) { section = line.upToFirstOccurrenceOf("]", true, false).toUpperCase(); continue; }
            // A space is data in a separator rule, not formatting to trim.
            if (section == "[VCPAD]" && raw.isNotEmpty())
            { if (!legacyPad) legacyPad = line.isEmpty() ? raw : line; continue; }
            if (line.isEmpty()) continue;
            if (section == "[ENDTYPE]" || section == "[ENDTYPE1]")
            {
                auto& target = section == "[ENDTYPE]" ? legacyEnd : legacyEnd1;
                if (!target) target = line;
                continue;
            }
            if (section == "[PRIORITY]")
            {
                auto items = juce::StringArray::fromTokens(line, ",", "");
                items.trim(); items.removeEmptyStrings(); rules.priorities.addArray(items);
                continue;
            }
            if (section == "[ALIAS]" && raw.containsChar('='))
            {
                auto key = raw.upToFirstOccurrenceOf("=", false, false).trim().toUpperCase();
                const auto value = raw.fromFirstOccurrenceOf("=", false, false);
                if (key == "VCPAD") rules.vcPad = value;
                else if (key == "VCVPAD") rules.vcvPad = value;
                else if (rules.aliasFormats.count(key)) rules.aliasFormats[key] = value.trim();
                continue;
            }
            auto fields = juce::StringArray::fromTokens(line, "=", "");
            fields.trim();
            if (section == "[REPLACE]" && fields.size() >= 2)
                rules.replacements[fields[0]] = fields[1];
            else if ((section == "[VOWEL]" && fields.size() >= 3)
                || (section == "[CONSONANT]" && fields.size() >= 2))
            {
                auto& table = section == "[VOWEL]" ? rules.vowels : rules.consonants;
                auto sounds = juce::StringArray::fromTokens(fields[section == "[VOWEL]" ? 2 : 1], ",", "");
                sounds.trim(); sounds.removeEmptyStrings();
                for (const auto& sound : sounds)
                {
                    table[sound] = fields[0];
                    if (section == "[VOWEL]") rules.vowelNames[sound] = fields[1];
                }
            }
        }
        // Legacy section precedence is independent of file order.
        if (legacyPad) rules.vcPad = *legacyPad;
        if (legacyEnd1 || legacyEnd) rules.aliasFormats["ENDING1"] = legacyEnd1 ? *legacyEnd1 : *legacyEnd;
        return rules;
    }
    juce::String normalize(juce::String lyric) const
    {
        lyric = hachi::lyricInPinyin(lyric.trim());
        if (const auto it = replacements.find(lyric); it != replacements.end()) return it->second;
        return lyric;
    }
    static juce::String lookup(const std::map<juce::String, juce::String>& table, const juce::String& key)
    {
        const auto it = table.find(key);
        if (it != table.end()) return it->second;
        juce::String equivalent;
        for (const auto& variant : spellingVariants(key))
            if (const auto found = table.find(variant); found != table.end())
            {
                // Conflicting explicit groups are not safe to infer across.
                if (equivalent.isNotEmpty() && equivalent != found->second) return key;
                equivalent = found->second;
            }
        return equivalent.isNotEmpty() ? equivalent : key;
    }

    juce::StringArray aliases(const juce::String& kind, const juce::String& previousVowel,
                              const juce::String& lyric) const
    {
        juce::StringArray result;
        for (auto pattern : juce::StringArray::fromTokens(aliasFormats.at(kind), ",", ""))
        {
            pattern = pattern.trim();
            juce::String value;
            bool valid = true;
            for (int pos = 0; pos < pattern.length();)
            {
                if (pattern[pos] != '%') { value += pattern[pos++]; continue; }
                const auto end = pattern.indexOfChar(pos + 1, '%');
                if (end < 0) { valid = false; break; }
                const auto token = pattern.substring(pos, end + 1);
                if (token == "%v%") value += previousVowel;
                else if (token == "%V%") value += lookup(vowelNames, lyric);
                else if (token == "%c%") value += lookup(consonants, lyric);
                else if (token == "%CV%") value += lyric;
                else if (token.equalsIgnoreCase("%VCPAD%")) value += vcPad;
                else if (token.equalsIgnoreCase("%VCVPAD%")) value += vcvPad;
                else { valid = false; break; }
                pos = end + 1;
            }
            if (valid && value.isNotEmpty()) result.addIfNotAlreadyThere(value);
        }
        return result;
    }
};

struct ChineseCvvcPlan
{
    UtauRenderRequest request;
    std::vector<std::size_t> owners;
    std::vector<double> offsets;
    std::vector<std::vector<UtauPhonemeSpan>> spans;
    juce::String warning;
};

// The VC's nominal advance is separate from its own OTO lead-in. Both the
// drag preview and the phonemizer use this rule, including its duration caps.
inline double cvvcTransitionDuration(const UtauSampleTiming& cv, double beat,
    std::optional<double> previousDuration, int velocity)
{
    auto duration = cv.preutteranceSeconds;
    if (cv.overlapSeconds == 0 && duration < beat / 4) duration = std::min(beat / 4, duration * 2);
    if (cv.overlapSeconds < 0) duration -= cv.overlapSeconds;
    const auto scaled = std::max(beat / 16, duration * consonantVelocityScale(velocity));
    return std::min(previousDuration ? *previousDuration / 1.5 : std::max(beat / 16, duration * 2), scaled);
}

inline ChineseCvvcPlan planChineseCvvc(const UtauRenderRequest& source)
{
    ChineseCvvcPlan plan;
    plan.request = source;
    plan.request.chineseCvvc = false; // also prevents recursion through the shared render adapter
    plan.request.cvvcPhonemes = true;
    plan.spans.resize(source.notes.size());
    juce::String error;
    const auto presamp = source.voicebankDirectory.getChildFile("presamp.ini");
    auto text = LegacyTextCodec::read(presamp, error);
    // ASCII oto.ini can be detected as UTF-8 even when the accompanying Chinese
    // presamp is GBK. Respect explicit encoding settings; otherwise retry GBK.
    if (!text && LegacyTextCodec::directoryCodePage(source.voicebankDirectory) == 0)
        text = LegacyTextCodec::read(presamp, error, 936);
    auto rules = text ? ChineseCvvcRules::parse(text->text) : ChineseCvvcRules{};
    if (rules.vowels.empty() || rules.consonants.empty())
    {
        plan.warning = "ZH CVVC: presamp.ini requires [VOWEL] and [CONSONANT]; using manual aliases";
        for (std::size_t i = 0; i < source.notes.size(); ++i) { plan.owners.push_back(i); plan.offsets.push_back(0); }
        return plan;
    }
    plan.request.notes.clear();
    struct Event { std::size_t owner; juce::String alias, kind; double start; float midi; bool base; };
    std::vector<Event> events;
    std::vector<std::size_t> order(source.notes.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) { return source.notes[a].startSeconds < source.notes[b].startSeconds; });
    const auto mapped = [&](const juce::String& alias, float midi)
    { return UtauRenderer::mappedAliasTiming(source.voicebankDirectory, alias, midi); };
    const auto mappedRule = [&](const juce::String& kind, const juce::String& vowel,
                                const juce::String& lyric, float midi)
        -> decltype(mapped(lyric, midi))
    {
        for (const auto& alias : rules.aliases(kind, vowel, lyric))
            if (auto timing = mapped(alias, midi)) return timing;
        return {};
    };
    std::vector<juce::String> lyrics;
    for (const auto& note : source.notes) lyrics.push_back(rules.normalize(note.alias));
    const auto adjacent = [&](std::size_t a, std::size_t b)
    {
        return !isRestLyric(source.notes[a].alias) && !isRestLyric(source.notes[b].alias)
            && source.notes[a].alias.isNotEmpty() && source.notes[b].alias.isNotEmpty()
            && std::abs(source.notes[a].startSeconds + source.notes[a].durationSeconds
                        - source.notes[b].startSeconds) < 0.002;
    };
    for (std::size_t position = 0; position < order.size(); ++position)
    {
        const auto i = order[position];
        const auto& note = source.notes[i];
        if (source.cancelled && source.cancelled()) break;
        auto lyric = lyrics[i];
        const bool hasPrev = position > 0 && adjacent(order[position - 1], i);
        const bool hasNext = position + 1 < order.size() && adjacent(i, order[position + 1]);
        const auto prev = hasPrev ? order[position - 1] : i;
        const auto prevVowel = hasPrev ? ChineseCvvcRules::lookup(rules.vowels, lyrics[prev]) : juce::String("-");
        const auto add = [&](juce::String alias, const char* kind, double offset, float midi, bool base)
        { events.push_back({ i, std::move(alias), kind, note.startSeconds + offset, midi, base }); };
        if (isRestLyric(note.alias) || note.alias.trim().isEmpty())
        { add(note.alias, "rest", 0, note.midiNote, true); continue; }
        // Space-containing aliases and [alias] are explicit overrides. They are not re-phonemized.
        if (lyric.containsChar(' ') || (lyric.startsWithChar('[') && lyric.endsWithChar(']')))
        {
            if (lyric.startsWithChar('[') && lyric.endsWithChar(']')) lyric = lyric.substring(1, lyric.length() - 1).trim();
            const auto oto = mapped(lyric, note.midiNote);
            add(oto ? oto->resolvedAlias : lyric, "manual", 0, note.midiNote, true);
            continue;
        }
        if (lyric == "-" || lyric.equalsIgnoreCase("R"))
        {
            const auto oto = mappedRule("ENDING1", prevVowel, hasPrev ? lyrics[prev] : lyric, note.midiNote);
            add(oto ? oto->resolvedAlias : juce::String("RR"), "release", 0, note.midiNote, true);
            continue;
        }
        const auto fullKind = hasPrev ? "VCV" : "BEGINING_CV";
        const auto contextual = [&](const juce::String& syllable)
            -> decltype(mapped(syllable, note.midiNote))
        {
            if (hasPrev && rules.priorities.contains(syllable)) return {};
            return mappedRule(fullKind, prevVowel, syllable, note.midiNote);
        };
        auto full = contextual(lyric);
        auto cv = mappedRule("CV", prevVowel, lyric, note.midiNote);
        if (!full && !cv && rules.replacements.find(hachi::lyricInPinyin(note.alias.trim())) == rules.replacements.end())
        {
            // Exact contextual and CV aliases win. Only choose an equivalent
            // spelling supported by this bank's rules and actual mapped OTO.
            for (const auto& variant : ChineseCvvcRules::spellingVariants(lyric))
            {
                if (!rules.vowels.count(variant) && !rules.consonants.count(variant)) continue;
                auto candidateFull = contextual(variant);
                auto candidateCv = mappedRule("CV", prevVowel, variant, note.midiNote);
                if (candidateFull || candidateCv)
                {
                    lyric = lyrics[i] = variant;
                    full = std::move(candidateFull); cv = std::move(candidateCv);
                    break;
                }
            }
        }
        if (!full && !cv)
        {
            // Keep the failed lyric distinct from an authored rest so the
            // renderer can audition it with the same missing-sample piano.
            add(lyric, "missing", 0, note.midiNote, true);
            if (plan.warning.length() < 1000) plan.warning += "Missing CV: " + lyric + "; ";
            continue;
        }
        if (!full)
        {
            const auto vcMidi = hasPrev ? source.notes[prev].midiNote : note.midiNote;
            const auto vc = mappedRule("VC", prevVowel, lyric, vcMidi);
            if (vc)
            {
                const auto beat = 60.0 / std::max(1.0, note.bpm > 0 ? note.bpm : source.bpm);
                const auto duration = cvvcTransitionDuration(*cv, beat,
                    hasPrev ? std::optional<double>(source.notes[prev].durationSeconds) : std::nullopt,
                    note.consonantVelocity);
                if (duration > 1.e-5) add(vc->resolvedAlias, "VC", -duration, vcMidi, false);
            }
        }
        add(full ? full->resolvedAlias : cv->resolvedAlias, full ? "VCV" : "CV", 0, note.midiNote, true);
        if (!hasNext)
        {
            const auto release = mappedRule("ENDING1", ChineseCvvcRules::lookup(rules.vowels, lyric), lyric, note.midiNote);
            const auto beat = 60.0 / std::max(1.0, note.bpm > 0 ? note.bpm : source.bpm);
            const auto tail = std::min(note.durationSeconds / 6, beat / 8);
            if (release && tail > 1.e-5) add(release->resolvedAlias, "release", note.durationSeconds - tail, note.midiNote, false);
        }
    }
    std::stable_sort(events.begin(), events.end(), [](const auto& a, const auto& b) { return a.start < b.start; });
    // Move only internal dividers. The first onset and the group's end stay
    // fixed, so adjacent phonemes exchange duration without moving score notes.
    // Work in event order even for stale overrides after a lyric/tempo edit.
    for (std::size_t first=0; first<events.size();)
    {
        auto last=first+1;
        while (last<events.size() && events[last].owner==events[first].owner) ++last;
        const auto& parent=source.notes[events[first].owner];
        auto end=parent.startSeconds+parent.durationSeconds;
        if (last<events.size()) end=std::min(end,events[last].start);
        std::vector<double> starts;std::vector<int> parts;
        for(auto i=first;i<last;++i){starts.push_back(events[i].start-parent.startSeconds);parts.push_back(cvvcEnvelopePart(events[i].kind));}
        applyCvvcBoundaries(starts,parts,parent.durationSeconds,end-parent.startSeconds,parent.cvvcBoundaries);
        for(auto i=first;i<last;++i)events[i].start=parent.startSeconds+starts[i-first];
        first=last;
    }
    std::vector<double> onset(source.notes.size(), 0);
    for (const auto& event : events)
    {
        const auto& parent = source.notes[event.owner];
        auto timing = mapped(event.alias, event.midi);
        if (event.base && parent.oto.enabled)
            timing = UtauRenderer::sampleTiming(source.voicebankDirectory, event.alias, event.midi,
                100, false, false, &parent.oto);
        auto pre = timing ? scaledUtauPreutterance(*timing, parent.consonantVelocity) : 0.0;
        if (event.base && parent.preutteranceOverrideEnabled) pre = parent.preutteranceSeconds;
        onset[event.owner] = std::min(onset[event.owner], event.start - parent.startSeconds - pre);
    }
    for (std::size_t e = 0; e < events.size(); ++e)
    {
        const auto& event = events[e];
        const auto& parent = source.notes[event.owner];
        auto piece = parent;
        const auto offset = event.start - parent.startSeconds;
        auto end = parent.startSeconds + parent.durationSeconds;
        if (e + 1 < events.size()) end = std::min(end, events[e + 1].start);
        if (end <= event.start + 1.e-6) continue;
        piece.alias = event.alias;
        piece.aliasIsResolved = event.kind != "manual" && event.kind != "rest";
        piece.startSeconds = event.start;
        piece.durationSeconds = end - event.start;
        piece.midiNote = event.midi;
        piece.preserveEnvelopeTiming = true;
        // Preserve an explicitly drawn parent curve in its original clock.
        // Unedited phonemes receive their own standard envelope below.
        piece.amplitudeEnvelope = UtauRenderer::fitAmplitudeEnvelope(parent.amplitudeEnvelope,
            -onset[event.owner], parent.durationSeconds);
        for (auto& point : piece.amplitudeEnvelope) point.timeSeconds -= offset;
        const auto midiShift = (parent.midiNote - piece.midiNote) * 100.0f;
        if (parent.timelinePitchCents)
            piece.timelinePitchCents = [curve = parent.timelinePitchCents, offset, midiShift](double t) { return curve(t + offset) + midiShift; };
        for (auto& point : piece.pitchCurve) { point.timeSeconds -= offset; point.cents += midiShift; }
        for (auto& curve : piece.flagCurves) for (auto& point : curve.second) point.first -= offset;
        if (!event.base)
        {
            piece.oto = {};
            piece.stpSeconds = 0;
            piece.preutteranceOverrideEnabled = piece.overlapOverrideEnabled = false;
        }
        const auto hasEarlier = std::any_of(events.begin(), events.begin() + static_cast<std::ptrdiff_t>(e),
            [&](const auto& other) { return other.owner == event.owner; });
        const auto hasLater = std::any_of(events.begin() + static_cast<std::ptrdiff_t>(e + 1), events.end(),
            [&](const auto& other) { return other.owner == event.owner; });
        if (hasLater) piece.tailFadeMode = 0;
        if (hasEarlier) piece.tailFadeSettings.head.mode = 0;
        const auto role = cvvcEnvelopePart(event.kind);
        auto timing = mapped(event.alias, event.midi);
        if (piece.oto.enabled)
            timing = UtauRenderer::sampleTiming(source.voicebankDirectory, event.alias, event.midi,
                100, false, false, &piece.oto);
        const auto lead = std::min(std::max(0.0, piece.startSeconds), std::max(0.0,
            piece.preutteranceOverrideEnabled ? piece.preutteranceSeconds
            : timing ? scaledUtauPreutterance(*timing, piece.consonantVelocity) : 0.0));
        if(role>0 && parent.amplitudeEnvelope.empty())
        {
            piece.cvvcStandardEnvelope=true;
            piece.amplitudeEnvelope.clear();
            for(const auto& p:standardCvvcEnvelope(piece.durationSeconds+lead))
                piece.amplitudeEnvelope.push_back({-lead+p.position*(piece.durationSeconds+lead),
                    envelopeDbFromGain(envelopeGainFromDb(p.value)*parent.cvvcDefaultEnvelopeGain),true});
        }
        if (const auto edits = parent.cvvcEnvelopes.find(role); edits != parent.cvvcEnvelopes.end())
            for (const auto& [key, points] : edits->second)
            {
                if (key.isEmpty())
                {
                    piece.cvvcStandardEnvelope=isStandardCvvcEnvelope(points,piece.durationSeconds+lead);
                    piece.amplitudeEnvelope.clear();
                    for (const auto& p : points) piece.amplitudeEnvelope.push_back(
                        { -lead + p.position * (piece.durationSeconds + lead), p.value, p.linear, p.bezier });
                    // An explicit per-piece envelope is the complete loudness shape.
                    piece.tailFadeMode = 0; piece.tailFadeSettings.head.mode = 0;
                }
                else
                {
                    auto curve = std::find_if(piece.flagCurves.begin(), piece.flagCurves.end(),
                        [&](const auto& value) { return value.first == key; });
                    if (curve == piece.flagCurves.end())
                    { piece.flagCurves.emplace_back(key, std::vector<std::pair<double,double>>{}); curve = std::prev(piece.flagCurves.end()); }
                    curve->second.clear();
                    for (const auto& p : points) curve->second.emplace_back(
                        -lead + p.position * (piece.durationSeconds + lead), p.value);
                }
            }
        piece.cvvcEnvelopes.clear();
        piece.cvvcBoundaries.clear();
        if(piece.tailFadeMode!=0||piece.tailFadeSettings.head.mode!=0)piece.cvvcStandardEnvelope=false;
        plan.spans[event.owner].push_back({ event.alias, event.kind, offset, end - parent.startSeconds, role, lead });
        plan.owners.push_back(event.owner);
        plan.offsets.push_back(offset);
        plan.request.notes.push_back(std::move(piece));
    }
    return plan;
}

// Both renderers use this adapter. Generated pieces are folded back into their
// original note for waveform callbacks, so editing IDs and project notes never split.
template <typename Render>
UtauRenderResult renderChineseCvvc(const UtauRenderRequest& source, Render&& render)
{
    auto plan = planChineseCvvc(source);
    struct Captured { std::size_t owner; double start, rate; juce::AudioBuffer<float> audio; };
    std::vector<Captured> pieces;
    plan.request.notePhonemes = {};
    if (source.notePhonemes || source.notePiece)
        plan.request.noteMixEnvelope = [&](std::size_t index, const UtauMixEnvelope& envelope)
    {
        if (index >= plan.owners.size()) return;
        auto& spans = plan.spans[plan.owners[index]];
        const auto found = std::find_if(spans.begin(), spans.end(), [&](const auto& span) {
            return std::abs(span.startSeconds - plan.offsets[index]) < 1.e-8;
        });
        if (found == spans.end()) return;
        found->mixEnvelope = envelope;
        found->mixEnvelope.startSeconds += plan.offsets[index];
    };
    else plan.request.noteMixEnvelope = {};
    if (source.notePiece)
        plan.request.notePiece = [&](std::size_t index, const juce::AudioBuffer<float>& audio, double rate,
            double lead, const std::function<float(double)>& gain, const std::function<float(double)>&)
        {
            if (index >= plan.owners.size() || rate <= 0) return;
            Captured piece { plan.owners[index], plan.offsets[index] - lead, rate, audio };
            for (int s = 0; s < piece.audio.getNumSamples(); ++s)
                for (int c = 0; c < piece.audio.getNumChannels(); ++c)
                    piece.audio.setSample(c, s, piece.audio.getSample(c, s) * (gain ? gain(s / rate - lead) : 1.0f));
            pieces.push_back(std::move(piece));
        };
    auto result = render(plan.request);
    if (source.cancelled && source.cancelled()) return result;
    for (std::size_t owner = 0; owner < source.notes.size(); ++owner)
    {
        if (source.notePhonemes) source.notePhonemes(owner, plan.spans[owner]);
        double first = 0, last = 0, rate = 0;
        for (const auto& piece : pieces) if (piece.owner == owner)
        { first = std::min(first, piece.start); last = std::max(last, piece.start + piece.audio.getNumSamples() / piece.rate); rate = piece.rate; }
        if (!source.notePiece || rate <= 0 || last <= first) continue;
        juce::AudioBuffer<float> combined(2, static_cast<int>(std::ceil((last - first) * rate)));
        combined.clear();
        for (const auto& piece : pieces) if (piece.owner == owner)
        {
            const auto start = static_cast<int>(std::lround((piece.start - first) * rate));
            const auto count = std::min(combined.getNumSamples() - start, piece.audio.getNumSamples());
            for (int c = 0; c < 2; ++c) combined.addFrom(c, start, piece.audio, std::min(c, piece.audio.getNumChannels() - 1), 0, count);
        }
        source.notePiece(owner, combined, rate, -first, [](double) { return 1.0f; }, [](double) { return 1.0f; });
    }
    if (plan.warning.isNotEmpty()) result.warning += (result.warning.isEmpty() ? "" : "; ") + plan.warning;
    result.backend += "+zh-cvvc";
    return result;
}
}
