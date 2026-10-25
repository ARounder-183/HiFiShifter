#pragma once
#include "../MaterialPronunciationCatalog.h"
#include "../OttoMaterialMigration.h"
#include "../OttoMaterialIO.h"
#include <juce_audio_formats/juce_audio_formats.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>

namespace hachi
{
// Compiled smoke entry, intentionally not wired by this module. Fixtures alone
// are not a runtime pass; parent registers Main/CMake and actually executes it.
inline bool runMaterialPronunciationCatalogSmoke(const juce::File& output)
{
    using C = MaterialPronunciationCatalog;
    const auto base = output.getChildFile("pronunciation-catalog-" + juce::Uuid().toString());
    if (base.createDirectory().failed()) return false;
    bool ok = true;
    juce::String log, error;
    const auto check = [&](const char* name, bool passed)
    {
        ok &= passed;
        log += juce::String(name) + "=" + (passed ? "PASS" : "FAIL") + "\n";
        std::cout << name << '=' << passed << std::endl;
    };
    const auto tone = [](const juce::File& audio)
    {
        if (audio.getParentDirectory().createDirectory().failed()) return false;
        juce::AudioBuffer<float> pcm(1, 48000);
        for (int i = 0; i < pcm.getNumSamples(); ++i)
            pcm.setSample(0, i, static_cast<float>(.1 * std::sin(i * .03)));
        juce::WavAudioFormat format;
        auto stream = audio.createOutputStream();
        if (!stream) return false;
        std::unique_ptr<juce::AudioFormatWriter> writer(
            format.createWriterFor(stream.release(), 48000, 1, 16, {}, 0));
        return writer && writer->writeFromAudioSampleBuffer(pcm, 0, pcm.getNumSamples());
    };
    const auto text = [](const juce::File& file, const juce::String& value)
    { return file.getParentDirectory().createDirectory().wasOk() && file.replaceWithText(value); };
    std::map<juce::String, juce::MemoryBlock> baseline;
    const auto remember = [&](const juce::File& file)
    {
        juce::MemoryBlock bytes;
        if (!file.loadFileAsData(bytes)) return false;
        baseline[file.getFullPathName()] = bytes; return true;
    };
    const auto unchanged = [&]()
    {
        for (const auto& [path, bytes] : baseline)
        { juce::MemoryBlock now; if (!juce::File(path).loadFileAsData(now) || now != bytes) return false; }
        return true;
    };
    const auto native = base.getChildFile("native");
    const auto other = base.getChildFile("other");
    const auto classic = base.getChildFile("classic-unexpanded");
    const auto blocked = base.getChildFile("blocked");
    const auto invalid = base.getChildFile("invalid");
    const auto a = native.getChildFile("a.wav");
    const auto b = native.getChildFile("deep/b.wav");
    const auto c = other.getChildFile("c.wav");
    const auto oldA = classic.getChildFile("unexpanded/a.wav");
    const auto oldB = classic.getChildFile("unexpanded/b.wav");
    const auto blockedAudio = blocked.getChildFile("unlisted.wav");
    const auto originalRoots = SampleSettings::materialRoots();
    check("fixture_audio", tone(a) && tone(b) && tone(c) && tone(oldA) && tone(oldB) && tone(blockedAudio));
    const juce::String nativeText = "[otto]\nversion=1\nunit=ms\n"
        "[aSame:a.wav]\npSame=same;escaped%3B%7C%3D%25|100,600|100,80,20|C:200,V:600\n"
        "[aOther:deep/b.wav]\npSame=same|100,600|100,80,20|C:200,V:600|flags:breath\n";
    check("fixture_manifests", text(native.getChildFile("otto.ini"), nativeText)
        && text(other.getChildFile("otto.ini"), "[otto]\nversion=1\nunit=ms\n[aSame:c.wav]\n"
                    "pSame=same|100,600|100,80,20|C:200,V:600\n")
        && text(blocked.getChildFile("otto.ini"), "[otto]\nversion=1\nunit=ms\n")
        && text(blocked.getChildFile("oto.ini"), "absent.wav=blocked,0,100,-500,80,20\nunlisted.wav=also-blocked,0,100,-500,80,20\n")
        && text(invalid.getChildFile("otto.ini"), "[otto]\nversion=999\nunit=ms\n")
        && text(invalid.getChildFile("oto.ini"), "absent.wav=invalid-fallback,0,100,-500,80,20\n"));
    const auto oto = classic.getChildFile("unexpanded/oto.ini");
    const juce::String classicText = "#Charset:UTF-8\na.wav=same,100,100,-500,80,20\n"
        "a.wav=secondary,100,100,-500,80,20\nb.wav=same,100,100,400,80,20\n"
        "b.wav=,100,100,-500,80,20\n";
    check("classic_fixture", text(oto, classicText));
    for (const auto& file : {a, b, c, oldA, oldB, blockedAudio, oto, native.getChildFile("otto.ini"),
                            other.getChildFile("otto.ini"), blocked.getChildFile("otto.ini"),
                            blocked.getChildFile("oto.ini"), invalid.getChildFile("otto.ini"),
                            invalid.getChildFile("oto.ini")})
        check("baseline_captured", remember(file));
    const std::vector<C::Root> roots{{native}, {other}, {classic}, {blocked}};
    const auto result = C::index(roots);
    check("catalog_read_only_success", result.ok() && result.entries.size() == 7 && unchanged());
    int same = 0, escaped = 0, old = 0;
    bool authoritative = true, fullRowsPreserved = true;
    C::Entry stale;
    for (const auto& entry : result.entries)
    {
        same += entry.alias == "same";
        escaped += entry.alias == "escaped;|=%";
        old += entry.format == C::Format::classicOto;
        if (entry.format == C::Format::classicOto) stale = entry;
        C::Entry resolved;
        std::vector<SampleRegionSetting> allRows;
        const bool resolvedOk = C::resolve(entry, resolved, error, &allRows);
        authoritative &= resolvedOk && resolved.audioId == entry.audioId && resolved.regionId == entry.regionId
            && resolved.row.regionStartSeconds == entry.row.regionStartSeconds
            && resolved.row.regionEndSeconds == entry.row.regionEndSeconds;
        if (entry.audio == oldA) fullRowsPreserved &= allRows.size() == 2;
    }
    check("same_alias_keeps_five_sources_and_rows", same == 5);
    check("escaped_secondary_alias_and_unexpanded_classic", escaped == 1 && old == 3);
    check("selected_ids_bounds_match_authoritative_reader", authoritative && fullRowsPreserved);
    check("root_only_empty_manifest_blocks_missing_audio_oto", C::index({{blocked}}).entries.empty()
          && C::index({{blocked}}).ok());
    std::vector<SampleRegionSetting> blockedRows;
    check("missing_otto_audio_section_authoritatively_empty", OttoMaterialMigration::readAudioAnnotations(
          blockedAudio, blocked, UtauMode::classic, blockedRows, error) && blockedRows.empty());
    const auto bad = C::index({{invalid}, {other}});
    check("invalid_manifest_rejects_no_fallback_other_roots_survive", !bad.ok() && !bad.errors.isEmpty()
          && bad.entries.size() == 1 && bad.entries.front().root == other);
    auto duplicates = roots; duplicates.push_back({native}); duplicates.push_back({classic});
    std::reverse(duplicates.begin(), duplicates.end());
    const auto dedup = C::index(duplicates);
    bool sameOrder = dedup.entries.size() == result.entries.size();
    if (sameOrder)
        for (std::size_t i = 0; i < result.entries.size(); ++i)
            sameOrder &= dedup.entries[i].audio == result.entries[i].audio
                && dedup.entries[i].regionId == result.entries[i].regionId
                && dedup.entries[i].alias == result.entries[i].alias;
    check("duplicate_roots_have_stable_identity_and_order", dedup.ok() && sameOrder);
    const auto conflict = C::index({{classic, UtauMode::classic}, {classic, UtauMode::jie}, {other}});
    check("conflicting_duplicate_mode_rejects_only_that_root", !conflict.ok() && conflict.entries.size() == 1
          && conflict.entries.front().root == other);
    check("cancelled_not_partial_success", C::index(roots, [] { return true; }).cancelled
          && C::index(roots, [] { return true; }).entries.empty());
    check("catalog_and_resolve_never_changed_sources", unchanged()
          && !classic.getChildFile("otto.ini").exists()
          && SampleSettings::materialRoots() == originalRoots);

    // Deliberate fixture edits happen only after read-only checks.
    check("stale_fixture_change", text(oto, classicText + ";external change\n"));
    C::Entry sentinel; sentinel.alias = "unchanged-output";
    std::vector<SampleRegionSetting> sentinelRows(1); sentinelRows[0].name = "unchanged-output";
    check("stale_classic_fingerprint_rejected_transactionally", !C::resolve(stale, sentinel, error, &sentinelRows)
          && error.isNotEmpty() && sentinel.alias == "unchanged-output"
          && sentinelRows.size() == 1 && sentinelRows[0].name == "unchanged-output");
    const auto nested = native.getChildFile("deep");
    check("nested_fixture_manifest", text(nested.getChildFile("otto.ini"),
        "[otto]\nversion=1\nunit=ms\n[aNested:b.wav]\npNested=nested|100,600|100,80,20|C:200,V:600\n"));
    const auto nestedResult = C::index({{native}, {nested}, {nested}});
    int nestedRows = 0, parentRows = 0;
    for (const auto& entry : nestedResult.entries)
    {
        if (entry.audio == b) nestedRows += entry.root == nested && entry.alias == "nested";
        if (entry.audio == a) ++parentRows;
    }
    check("deepest_registered_root_owns_audio_without_collision", nestedResult.ok()
          && nestedRows == 1 && parentRows == 2 && nestedResult.entries.size() == 3);
    // Explicit negative cutoff can index without a header, but physical range
    // must be rejected by authoritative selection rather than silently clamped.
    const auto beyond = base.getChildFile("beyond");
    check("beyond_fixture", tone(beyond.getChildFile("a.wav"))
          && text(beyond.getChildFile("oto.ini"), "a.wav=beyond,0,100,-2000,80,20\n"));
    const auto pending = C::index({{beyond}});
    check("physical_end_revalidated_at_selection", pending.ok() && pending.entries.size() == 1
          && !C::resolve(pending.entries.front(), sentinel, error));
    const auto report = base.getChildFile("verification.txt");
    return report.replaceWithText(log) && ok;
}
}
