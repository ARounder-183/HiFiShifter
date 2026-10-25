#pragma once
#include "../PronunciationTiming.h"
#include "../SampleSettings.h"
#include "../AudioEngine.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <set>

namespace hachi
{
inline bool runPronunciationTimingSmoke(const juce::File& output)
{
    if (!output.createDirectory()) return false;
    bool ok = true; int checks = 0; juce::Array<juce::var> results;
    const auto check = [&](const char* name, bool pass)
    {
        ok &= pass; ++checks; std::cout << name << '=' << pass << std::endl;
        auto item = new juce::DynamicObject(); item->setProperty("name", name); item->setProperty("pass", pass);
        results.add(juce::var(item));
    };
    const auto near = [](double a, double b) { return std::abs(a - b) < 1.e-8; };
    SampleRegionSetting source; source.name = "ka"; source.regionStartSeconds = 2; source.regionEndSeconds = 3;
    source.alignmentSeconds = 2.12; source.fixedDurationSeconds = .2; source.overlapSeconds = -.35;
    source.provenance = "user"; source.confidence = .87f; source.relativePitchCents = 37;
    source.melodyneFormantCents = 81; source.melodyneAmplitude = .7; source.materialRegionId = "stable-region";
    source.amplitudeEnvelope = {{0, -2}, {1, -3}};
    check("simplified_consonant_defaults_off", !source.simplifiedConsonant);
    auto timing = pronunciationTimingFor(source);
    check("absolute_P_and_C_share_one_interpretation", near(timing.alignmentSeconds, 2.12) && near(timing.consonantEndSeconds, 2.2)
        && near(protectedPronunciationSourceEnd(timing), 2.2) && near(consonantAfterAlignmentSeconds(timing), .08));
    check("negative_overlap_is_preserved_as_signed_offset", near(timing.overlapSeconds, -.35));
    const auto segments = pronunciationSegmentsFor(source);
    check("C_ends_at_C_marker_not_P", segments.size() == 2 && segments[0].role == NativeSegmentRole::consonant
        && near(segments[0].sourceStartSeconds, 0) && near(segments[0].sourceEndSeconds, .2) && !segments[0].stretchable
        && segments[1].role == NativeSegmentRole::vowel && segments[1].stretchable);
    check("ordinary_segment_ids_remain_compact", segments.size() == 2 && segments[0].id == "segment_1" && segments[1].id == "segment_2");
    check("deriving_segments_does_not_rewrite_loaded_metadata", source.segments.empty()
        && near(source.regionStartSeconds, 2) && near(source.fixedDurationSeconds, .2) && !source.simplifiedConsonant);
    const auto delegated = SampleSettings::nativeSegmentsFor(source);
    check("SampleSettings_delegates_same_canonical_segments", delegated.size() == segments.size()
        && near(delegated[0].sourceEndSeconds, segments[0].sourceEndSeconds) && delegated[0].role == segments[0].role);
    std::vector<SourceTimePoint> map; juce::String error;
    check("one_to_two_second_map_constructs", makePronunciationStretchMap(source, 2, map, error));
    check("C_point20_stays_point20_only_V_point80_becomes1point80", map.size() == 3
        && near(map[0].targetSeconds, 0) && near(map[0].sourceSeconds, 2)
        && near(map[1].targetSeconds, .2) && near(map[1].sourceSeconds, 2.2)
        && near(map[2].targetSeconds, 2) && near(map[2].sourceSeconds, 3));
    auto earlyV = source; earlyV.fixedDurationSeconds = .1; earlyV.alignmentSeconds = 2.3;
    const auto early = pronunciationSegmentsFor(earlyV);
    check("C_before_P_has_fixed_early_vowel", early.size() == 3 && near(early[0].sourceEndSeconds, .1)
        && early[1].role == NativeSegmentRole::vowel && !early[1].stretchable && near(early[1].sourceEndSeconds, .3)
        && early[2].role == NativeSegmentRole::vowel && early[2].stretchable);
    check("P_protects_source_even_when_C_is_earlier", makePronunciationStretchMap(earlyV, 2, map, error)
        && map.size() == 4 && near(map[1].targetSeconds, .1) && near(map[2].targetSeconds, .3)
        && near(map[2].sourceSeconds, 2.3) && near(map[3].targetSeconds, 2));

    auto earlyLinked = earlyV;
    check("enabling_with_C_before_P_uses_zero_derived_gap", setSimplifiedConsonant(earlyLinked, true, -1, error)
        && near(pronunciationTimingFor(earlyLinked).consonantEndSeconds, earlyLinked.alignmentSeconds));
    auto linked = source;
    check("enable_simplified_sentinel_keeps_current_gap", setSimplifiedConsonant(linked, true, -1, error)
        && linked.simplifiedConsonant && near(pronunciationTimingFor(linked).consonantEndSeconds, 2.2));
    check("moving_P_preserves_derived_delta_when_enabled", editPronunciationAlignment(linked, 2.4, error)
        && near(pronunciationTimingFor(linked).consonantEndSeconds, 2.48));
    check("linked_C_cannot_move_before_P", editPronunciationConsonant(linked, 2.1, error)
        && near(pronunciationTimingFor(linked).consonantEndSeconds, 2.4));
    check("explicit_delta_updates_C_to_P_plus_delta", setSimplifiedConsonant(linked, true, .15, error)
        && near(pronunciationTimingFor(linked).consonantEndSeconds, 2.55));
    check("source_end_clamp_has_visible_derived_gap", editPronunciationAlignment(linked, 2.95, error)
        && near(pronunciationTimingFor(linked).consonantEndSeconds, 3)
        && near(consonantAfterAlignmentSeconds(pronunciationTimingFor(linked)), .05));
    check("subsequent_P_edit_uses_current_visible_gap", editPronunciationAlignment(linked, 2.9, error)
        && near(pronunciationTimingFor(linked).consonantEndSeconds, 2.95));
    check("disable_keeps_existing_C_position", setSimplifiedConsonant(linked, false, -1, error)
        && !linked.simplifiedConsonant && near(pronunciationTimingFor(linked).consonantEndSeconds, 2.95));
    check("off_C_and_P_are_independent", editPronunciationConsonant(linked, 2.05, error)
        && editPronunciationAlignment(linked, 2.3, error) && near(pronunciationTimingFor(linked).consonantEndSeconds, 2.05));
    check("independent_early_C_still_protects_later_P", near(protectedPronunciationSourceEnd(pronunciationTimingFor(linked)), 2.3));
    auto explicitEnable = source;
    check("explicit_delta_on_first_enable_is_honored", setSimplifiedConsonant(explicitEnable, true, .25, error)
        && near(pronunciationTimingFor(explicitEnable).consonantEndSeconds, 2.37));
    const auto beforeInvalid = pronunciationTimingFor(explicitEnable);
    check("invalid_delta_is_atomic", !setSimplifiedConsonant(explicitEnable, true, -2, error)
        && near(pronunciationTimingFor(explicitEnable).consonantEndSeconds, beforeInvalid.consonantEndSeconds));
    check("NaN_delta_is_rejected", !setSimplifiedConsonant(explicitEnable, true, std::numeric_limits<double>::quiet_NaN(), error));

    auto detailed = source; detailed.vowelSplitsExplicit = true;
    NativeSegment a; a.id = "v-a"; a.alias = "v1"; a.role = NativeSegmentRole::vowel;
    a.sourceStartSeconds = 2; a.sourceEndSeconds = 2.6; a.provenance = "imported"; a.confidence = .82f; a.stretchWeight = 1;
    NativeSegment b = a; b.id = "v-b"; b.alias = "v2"; b.sourceStartSeconds = 2.6; b.sourceEndSeconds = 2.9; b.stretchWeight = 3;
    NativeSegment tail = a; tail.id = "noise-tail"; tail.role = NativeSegmentRole::noise;
    tail.sourceStartSeconds = 2.9; tail.sourceEndSeconds = 3; tail.stretchable = true;
    detailed.segments = {a, b, tail};
    check("apply_normalizes_absolute_source_zones", applyPronunciationTiming(detailed, pronunciationTimingFor(detailed), error));
    const auto find = [](const auto& values, const juce::String& id) -> const NativeSegment*
    { for (const auto& value : values) if (value.id == id) return &value; return nullptr; };
    const auto* keptA = find(detailed.segments, "v-a"); const auto* keptB = find(detailed.segments, "v-b");
    const auto* keptTail = find(detailed.segments, "noise-tail");
    check("vowel_split_keeps_ID_alias_weight_and_provenance", keptA && keptB && near(keptA->sourceStartSeconds, 2.2)
        && near(keptA->sourceEndSeconds, 2.6) && keptA->alias == "v1" && keptA->provenance == "imported"
        && keptA->confidence == .82f && near(keptB->stretchWeight, 3));
    check("explicit_noise_tail_stays_fixed_without_disappearing", keptTail && keptTail->role == NativeSegmentRole::noise
        && !keptTail->stretchable && near(keptTail->sourceStartSeconds, 2.9) && near(keptTail->sourceEndSeconds, 3));
    check("multiple_vowels_use_duration_times_stretch_weight", makePronunciationStretchMap(detailed, 2, map, error)
        && map.size() == 5 && near(map[1].targetSeconds, .2) && near(map[2].targetSeconds, .2 + 1.7 * .4 / 1.3)
        && near(map[3].targetSeconds, 1.9) && near(map[4].targetSeconds, 2)
        && near(map[2].sourceSeconds, 2.6) && near(map[3].sourceSeconds, 2.9));
    check("row_identity_provenance_expression_and_signed_overlap_are_preserved", detailed.materialRegionId == source.materialRegionId
        && detailed.pronunciationKind == source.pronunciationKind && detailed.relativePitchCents == 37 && detailed.melodyneFormantCents == 81
        && near(detailed.melodyneAmplitude, .7) && detailed.amplitudeEnvelope.size() == 2 && detailed.provenance == "user"
        && near(detailed.overlapSeconds, -.35));
    auto arbitrary = detailed; for (auto& segment : arbitrary.segments) if (segment.id == "v-b") segment.role = NativeSegmentRole::consonant;
    const auto corrected = pronunciationSegmentsFor(arbitrary);
    const auto* correctedB = find(corrected, "v-b");
    check("source_C_line_overrides_old_arbitrary_CV_dropdown", correctedB && correctedB->role == NativeSegmentRole::vowel && correctedB->stretchable);
    auto breath = source; breath.pronunciationKind = PronunciationKind::breath;
    const auto breathParts = pronunciationSegmentsFor(breath);
    check("whole_breath_identity_does_not_erase_CV_source_facts", breathParts.size() == segments.size()
        && breathParts[0].role == NativeSegmentRole::consonant && breathParts[1].role == NativeSegmentRole::vowel);
    check("all_pronunciation_identities_roundtrip", parsePronunciationKind(pronunciationKindName(PronunciationKind::normal)) == PronunciationKind::normal
        && parsePronunciationKind(pronunciationKindName(PronunciationKind::transition)) == PronunciationKind::transition
        && parsePronunciationKind(pronunciationKindName(PronunciationKind::breath)) == PronunciationKind::breath);
    auto fullyFixed = source; fullyFixed.alignmentSeconds = fullyFixed.regionEndSeconds;
    check("fully_fixed_source_allows_identity_duration", makePronunciationStretchMap(fullyFixed, 1, map, error)
        && near(map.back().targetSeconds, 1) && near(map.back().sourceSeconds, 3));
    check("fully_fixed_source_rejects_other_durations", !makePronunciationStretchMap(fullyFixed, 2, map, error));
    map = {{7, 9}};
    check("impossible_contraction_is_atomic_no_consonant_stretch_or_vowel_drop", !makePronunciationStretchMap(earlyV, .25, map, error)
        && map.size() == 1 && near(map[0].targetSeconds, 7) && near(map[0].sourceSeconds, 9));
    check("zero_target_is_rejected", !makePronunciationStretchMap(source, 0, map, error));
    auto tiny = source; tiny.regionEndSeconds = 2.0005; tiny.alignmentSeconds = 2.0001; tiny.fixedDurationSeconds = .0002;
    check("valid_submillisecond_source_is_not_artificially_extended", applyPronunciationTiming(tiny, pronunciationTimingFor(tiny), error)
        && near(tiny.regionEndSeconds, 2.0005) && makePronunciationStretchMap(tiny, .001, map, error)
        && near(map[1].targetSeconds, .0002) && near(map.back().sourceSeconds, 2.0005));
    auto clamped = source; auto outside = pronunciationTimingFor(clamped); outside.alignmentSeconds = 99; outside.consonantEndSeconds = -3; outside.overlapSeconds = 5;
    check("markers_normalize_to_source_bounds_without_clamping_overlap", applyPronunciationTiming(clamped, outside, error)
        && near(clamped.alignmentSeconds, 3) && near(clamped.fixedDurationSeconds, 0) && near(clamped.overlapSeconds, 5));
    auto invalid = pronunciationTimingFor(source); invalid.sourceEndSeconds = invalid.sourceStartSeconds;
    const auto savedStart = source.regionStartSeconds;
    check("invalid_source_bounds_do_not_mutate_row", !applyPronunciationTiming(source, invalid, error)
        && near(source.regionStartSeconds, savedStart) && near(source.regionEndSeconds, 3));
    auto overlap = detailed; overlap.segments[1].sourceStartSeconds = 2.01;
    check("overlapping_source_zones_reject_instead_of_dropping_facts", !applyPronunciationTiming(overlap, pronunciationTimingFor(overlap), error));
    std::set<juce::String> ids; bool unique = true;
    for (const auto& segment : detailed.segments) unique &= segment.id.isNotEmpty() && ids.insert(segment.id).second;
    check("canonical_segment_ids_are_unique", unique);
    auto epochs = source; epochs.segments.clear();
    for (int i = 0; i < 1000; ++i)
    {
        NativeSegment epoch; epoch.id = "analysis_" + juce::String(i); epoch.alias = "analysis";
        epoch.role = i < 200 ? NativeSegmentRole::consonant : (i % 3 == 0 && i != 999 ? NativeSegmentRole::noise : NativeSegmentRole::vowel);
        epoch.sourceStartSeconds = 2 + i / 1000.0; epoch.sourceEndSeconds = 2 + (i + 1) / 1000.0;
        epoch.alignmentSeconds = 2.12; epoch.stretchWeight = 7;
        epochs.segments.push_back(epoch);
    }
    const auto compact = pronunciationSegmentsFor(epochs);
    check("one_thousand_analysis_epochs_are_not_authoring_regions", !epochs.vowelSplitsExplicit && compact.size() == 2
        && compact[0].role == NativeSegmentRole::consonant && compact[1].role == NativeSegmentRole::vowel && compact[1].stretchable);
    check("P_inside_C_is_marker_not_extra_C_region_boundary", compact.size() == 2 && near(compact[0].sourceEndSeconds, .2));
    check("default_coalescing_does_not_mutate_raw_analysis_or_scalars", epochs.segments.size() == 1000
        && epochs.segments.front().id == "analysis_0" && epochs.segments.back().id == "analysis_999"
        && epochs.segments.front().alias == "analysis" && near(epochs.alignmentSeconds, 2.12) && near(epochs.fixedDurationSeconds, .2));
    check("default_compact_projection_can_reuse_one_C_and_last_V_ID", compact.size() == 2
        && compact[0].id == "analysis_0" && compact[1].id == "analysis_999" && compact[1].alias == "ka" && near(compact[1].stretchWeight, 1));
    check("automatic_noise_epochs_do_not_fragment_default_stretch_map", makePronunciationStretchMap(epochs, 2, map, error)
        && map.size() == 3 && near(map[1].targetSeconds, .2) && near(map.back().sourceSeconds, 3));
    auto manual = source; manual.vowelSplitsExplicit = true;
    auto namedA = a; namedA.id = "named-a"; namedA.alias = "an"; namedA.sourceStartSeconds = 2.2; namedA.sourceEndSeconds = 2.6;
    auto namedB = namedA; namedB.id = "named-b"; namedB.sourceStartSeconds = 2.6; namedB.sourceEndSeconds = 3; namedB.stretchWeight = 3;
    manual.segments = {namedA, namedB}; const auto named = pronunciationSegmentsFor(manual);
    check("explicit_named_V_subdivisions_preserve_duplicate_names_distinct_IDs", named.size() == 3
        && find(named, "named-a") && find(named, "named-b") && find(named, "named-a")->alias == "an" && find(named, "named-b")->alias == "an"
        && near(find(named, "named-b")->stretchWeight, 3));
    auto unnamed = manual; for (auto& segment : unnamed.segments) segment.alias = "-";
    check("unnamed_V_epochs_are_not_manual_subdivision_boundaries", pronunciationSegmentsFor(unnamed).size() == 2);
    auto unlimited = source; unlimited.vowelSplitsExplicit = true; unlimited.segments.clear();
    for (int i = 0; i < 200; ++i)
    {
        auto segment = namedA; segment.id = "manual_" + juce::String(i); segment.alias = "repeat";
        segment.sourceStartSeconds = 2.2 + .8 * i / 200; segment.sourceEndSeconds = 2.2 + .8 * (i + 1) / 200;
        unlimited.segments.push_back(segment);
    }
    check("explicit_named_V_subdivisions_have_no_arbitrary_count_cap", pronunciationSegmentsFor(unlimited).size() == 201);
    auto earlyEpochs = epochs; earlyEpochs.alignmentSeconds = 2.3; earlyEpochs.fixedDurationSeconds = .1;
    const auto earlyCompact = pronunciationSegmentsFor(earlyEpochs);
    check("legacy_C_before_P_adds_only_required_fixed_early_V_interval", earlyCompact.size() == 3
        && earlyCompact[1].role == NativeSegmentRole::vowel && !earlyCompact[1].stretchable && earlyCompact[2].stretchable);
    {
        ProjectData project; TrackData track; track.id="timing-capability";track.pitchAlgorithm=PitchAlgorithm::nsfHifigan;track.nativeNsfAudio=true;
        ClipData clip;clip.id="timing-capability-clip";clip.durationSeconds=1;
        NoteData note;note.id="timing-capability-note";note.durationSeconds=1;
        note.materialAudioId="a1";note.materialRegionId="p1";note.materialSourceStartSeconds=note.nativeSourceStartSeconds=1;
        note.materialSourceEndSeconds=note.nativeSourceEndSeconds=2;
        note.utauPreutteranceOverrideEnabled=true;note.utauPreutteranceSeconds=.12;
        note.utauOverlapOverrideEnabled=true;note.utauOverlapSeconds=-.03;
        auto row=source;row.regionStartSeconds=1;row.regionEndSeconds=2;row.alignmentSeconds=1.12;
        row.fixedDurationSeconds=.2;row.overlapSeconds=-.03;row.vowelSplitsExplicit=false;
        note.nativeSegments=pronunciationSegmentsFor(row);clip.notes={note};track.clips={clip};project.tracks={track};
        check("canonical_material_P_overlap_are_native_geometry_not_unsupported_protocol",AudioEngine::renderCapabilityWarnings(project).isEmpty());
        project.tracks[0].clips[0].notes[0].utauPreutteranceSeconds=.18;
        check("independent_nonmatching_protocol_P_still_warns",!AudioEngine::renderCapabilityWarnings(project).isEmpty());
        project.tracks[0].clips[0].notes[0]=note;project.tracks[0].clips[0].notes[0].utauOverlapSeconds=.04;
        check("independent_nonmatching_protocol_overlap_still_warns",!AudioEngine::renderCapabilityWarnings(project).isEmpty());
        project.tracks[0].clips[0].notes[0]=note;project.tracks[0].clips[0].notes[0].materialRegionId.clear();
        check("unbound_protocol_overrides_are_not_silently_exempted",!AudioEngine::renderCapabilityWarnings(project).isEmpty());
        project.tracks[0].clips[0].notes[0]=note;project.tracks[0].clips[0].notes[0].utauFlags="Hb15";
        check("snapshot_timing_does_not_hide_unsupported_FLAGS",!AudioEngine::renderCapabilityWarnings(project).isEmpty());
        project.tracks[0].clips[0].notes[0]=note;auto& cropped=project.tracks[0].clips[0].notes[0];
        cropped.nativeSourceStartSeconds=1.3;for(auto& segment:cropped.nativeSegments)segment.alignmentSeconds=0;
        check("cropped_canonical_P_clamps_to_actual_source_window",AudioEngine::renderCapabilityWarnings(project).isEmpty());
    }
    check("timing_service_does_not_write_OTO_or_source_files", !output.getChildFile("oto.ini").exists()
        && !output.getChildFile("otto.ini").exists() && !output.getChildFile("source.wav").exists());
    auto report = new juce::DynamicObject(); report->setProperty("ok", ok); report->setProperty("checks", checks);
    report->setProperty("results", juce::var(results));
    const bool written = output.getChildFile("verification.json").replaceWithText(juce::JSON::toString(juce::var(report)));
    std::cout << "pronunciation_timing_ok=" << (ok && written) << "; checks=" << checks << std::endl;
    return ok && written;
}
}
