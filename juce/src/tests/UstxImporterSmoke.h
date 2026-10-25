#pragma once
#include "../backend/UstxImporter.h"
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

namespace hachi
{
// Small actual upstream dataset, not a copyrighted song/voicebank:
// OpenUtau.Test/Core/USTx/UstxYamlTest.cs, UNoteSerializationTest expected YAML
// at commit 33302b3d7cecf9ae830d4229639e06788f717b3c. The note below is verbatim;
// only its surrounding project is synthetic.
// https://github.com/openutau/OpenUtau/blob/33302b3d7cecf9ae830d4229639e06788f717b3c/OpenUtau.Test/Core/USTx/UstxYamlTest.cs
// https://github.com/openutau/OpenUtau/blob/33302b3d7cecf9ae830d4229639e06788f717b3c/LICENSE.txt
//
// The MIT License (MIT)
// Copyright (c) 2014 StAkira
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
inline bool ustxImporterSmoke(const juce::File& folder)
{
    bool ok = true; int checks = 0;
    juce::Array<juce::var> failures;
    const auto check = [&](const juce::String& name, bool pass)
    {
        ++checks; ok &= pass;
        if (!pass) failures.add(name);
        std::cout << "ustx_" << name << "=" << pass << std::endl;
    };
    using backend::UstxImporter;
    juce::String error; juce::StringArray warnings;
    const std::string upstreamNote = R"USTX(position: 120
duration: 60
tone: 42
lyric: あ
pitch:
  data:
  - {x: -5, y: 0, shape: io}
  - {x: 5, y: 0, shape: io}
  snap_first: true
vibrato: {length: 0, period: 175, depth: 25, in: 10, out: 10, shift: 0, drift: 0, vol_link: 0}
tuning: 0
phoneme_expressions:
- {index: 0, abbr: vel, value: 123}
phoneme_overrides: []
)USTX";
    std::string actualSample = "ustx_version: '0.6'\nresolution: 480\ntracks: [{track_name: Test}]\nvoice_parts:\n- track_no: 0\n  position: 0\n  duration: 480\n  notes:\n";
    std::istringstream lines(upstreamNote); std::string line; bool firstLine = true;
    while (std::getline(lines, line))
    {
        actualSample += firstLine ? "  - " : "    ";
        actualSample += line + "\n"; firstLine = false;
    }
    const auto upstream = UstxImporter::parse(actualSample, error, warnings);
    check("actual_upstream_note_parsed", upstream && upstream->notes.size() == 1);
    if (upstream && upstream->notes.size() == 1)
    {
        const auto& n = upstream->notes[0];
        check("upstream_utf8_lyric", n.lyric == juce::String::fromUTF8("あ"));
        check("upstream_score_units", n.noteNum == 42 && n.lengthTicks == 60
            && n.positionQuarterNotes == 0.25 && n.durationQuarterNotes == 0.125);
        check("upstream_control_units", n.hasPitchBend && n.pitchStartMs == -5
            && n.widthsMs.size() == 1 && n.widthsMs[0] == 10 && n.shapes[0].isEmpty());
        check("upstream_expression_retained_not_faked", upstream->sourceText.contains("abbr: vel, value: 123")
            && !n.velocity && warnings.joinIntoString("\n").contains("phoneme_expressions"));
    }
    const std::string synthetic = R"USTX(# Synthetic UTF-8/multipart/control test, not an external composition.
ustx_version: '0.10'
resolution: 480
name: '测试 🎵'
expressions:
  vel: {name: velocity, abbr: vel, type: Numerical, min: 0, max: 200, default_value: 100}
time_signatures:
- {bar_position: 0, beat_per_bar: 3, beat_unit: 4}
- {bar_position: 2, beat_per_bar: 5, beat_unit: 8}
tempos:
- {position: 960, bpm: 90.5}
- {position: 0, bpm: 120}
tracks:
- {track_name: '主旋律 あ', singer: '../../not-a-bank', phonemizer: Some.Plugin, volume: -3}
- {track_name: '和声', singer: absent}
voice_parts:
- name: first
  track_no: 0
  position: 480
  duration: 1920
  notes:
  - position: 120
    duration: 240
    tone: 60
    lyric: 'on'
    pitch:
      data:
      - {x: -20, y: -10, shape: l}
      - {x: 0, y: 2, shape: i}
      - {x: 20, y: 4, shape: o}
      - {x: 40, y: 0, shape: io}
      snap_first: false
    vibrato: {length: 50, period: 180, depth: 40, in: 15, out: 20, shift: 25, drift: -10, vol_link: 30}
    phoneme_expressions:
    - {index: 0, abbr: vel, value: 123.5}
    - {index: 1, abbr: vel, value: 90}
    phoneme_overrides: [{index: 0, phoneme: 'a', offset: 10}]
  - {position: 960, duration: 120, tone: 62, lyric: R}
  curves:
  - {abbr: pitd, xs: [-5, 0, 100], ys: [0, 50, 0]}
- {name: later, track_no: 0, position: 3840, duration: 960, notes: [{position: 240, duration: 480, tone: 64, lyric: '+あ'}]}
- track_no: 1
  position: -240
  duration: 1920
  notes:
  - {position: 0, duration: 480, tone: 48, lyric: 'null'}
  - position: 480
    duration: 480
    tone: 52
    lyric: 'true'
    pitch: {data: [{x: -10, y: 999, shape: io}, {x: 20, y: 0, shape: io}], snap_first: true}
wave_parts:
- {track_no: 1, position: 0, relative_path: '../../../../not-read.wav', file_duration_ms: 1000}
vendor_extension: {opaque: 'kept exactly'}
)USTX";
    warnings.clear();
    const auto song = UstxImporter::parse(synthetic, error, warnings);
    check("multitrack_multipart_parse", song && song->notes.size() == 3 && song->additionalTracks.size() == 1);
    if (song && song->notes.size() == 3 && song->additionalTracks.size() == 1)
    {
        const auto& notes = song->notes; const auto& harmony = song->additionalTracks[0].notes;
        check("gaps_and_offsets_not_sequentialized", notes[0].positionQuarterNotes == 1.25
            && notes[1].positionQuarterNotes == 3.0 && notes[2].positionQuarterNotes == 8.5);
        check("rest_keeps_its_actual_position", notes[1].isRest() && notes[1].durationQuarterNotes == 0.25);
        check("tempo_map_sorted_and_exact", song->tempoChanges.size() == 2 && song->tempo == 120
            && song->tempoChanges[0].quarterPosition == 0 && song->tempoChanges[1].quarterPosition == 2
            && song->tempoChanges[1].bpm == 90.5);
        check("meter_at_origin", song->numerator == 3 && song->denominator == 4);
        check("pitch_shapes_exact_mapping", notes[0].hasPitchBend && notes[0].shapes.size() == 3
            && notes[0].shapes[0] == "s" && notes[0].shapes[1] == "j" && notes[0].shapes[2] == "r"
            && notes[0].pitchStartTenths == -10 && notes[0].pitchTenths[0] == 2);
        check("vibrato_units", notes[0].hasVibrato && notes[0].vibratoLengthPercent == 50
            && notes[0].vibratoCycleMs == 180 && notes[0].vibratoDepthCents == 40
            && notes[0].vibratoPhasePercent == 25 && notes[0].vibratoOffsetPercent == -10);
        check("pickup_and_snap_first", harmony.size() == 2 && harmony[0].positionQuarterNotes == -0.5
            && harmony[1].pitchStartTenths == -40);
        check("singer_path_never_bound", song->voiceDirectory.isEmpty() && song->additionalTracks[0].voiceDirectory.isEmpty());
        const auto messages = warnings.joinIntoString("\n");
        check("unsupported_explicit_warning", messages.contains("time_signatures") && messages.contains("curves")
            && messages.contains("phoneme_expressions") && messages.contains("vol_link")
            && messages.contains("wave_parts") && messages.contains("vendor_extension"));
        check("raw_complete_utf8_roundtrip", std::string(song->sourceText.toRawUTF8(), song->sourceText.getNumBytesAsUTF8()) == synthetic);
        juce::String e; juce::StringArray w;
        const auto again = UstxImporter::parse(std::string(song->sourceText.toRawUTF8()), e, w);
        check("expression_and_curve_source_reparse", again && again->sourceText == song->sourceText
            && again->sourceText.contains("value: 123.5") && again->sourceText.contains("xs: [-5, 0, 100]")
            && again->notes[2].positionQuarterNotes == notes[2].positionQuarterNotes);
    }
    const std::string minimal = "{\"ustx_version\":\"0.6\",\"tracks\":[{\"track_name\":\"JSON\"}],\"voice_parts\":[{\"track_no\":0,\"position\":960,\"notes\":[{\"position\":120,\"duration\":60,\"tone\":42,\"lyric\":\"あ\"}]}]}";
    auto json = UstxImporter::parse(minimal, error, warnings);
    check("json_valid_yaml_same_semantics", json && json->notes.size() == 1 && json->notes[0].positionQuarterNotes == 2.25);
    const auto reject = [&](const juce::String& name, const std::string& input)
    {
        juce::String e; juce::StringArray w;
        check(name, !UstxImporter::parse(input, e, w) && e.isNotEmpty());
    };
    reject("malformed_yaml", "ustx_version: '0.6'\ntracks: [");
    reject("duplicate_mapping_keys", "ustx_version: '0.6'\ntracks: []\ntracks: [{}]");
    reject("alias_cycle", "ustx_version: '0.6'\ntracks: &a [*a]");
    reject("multiple_documents", actualSample + "\n---\n" + actualSample);
    reject("custom_tag", "ustx_version: '0.6'\ntracks: !object []");
    reject("invalid_utf8", actualSample + std::string("\xc0\xaf", 2));
    reject("embedded_nul", actualSample + std::string(1, '\0'));
    reject("escaped_unicode_surrogate", "ustx_version: '0.6'\ntracks: [{track_name: \"\\uD800\"}]");
    reject("no_format_version", "tracks: [{}]\nvoice_parts: []");
    reject("invalid_resolution", "ustx_version: '0.6'\nresolution: 960\ntracks: [{}]");
    reject("nonfinite_tempo", "ustx_version: '0.6'\ntracks: [{}]\ntempos: [{position: 0, bpm: .nan}]");
    reject("infinite_expression", "ustx_version: '0.6'\ntracks: [{}]\nexpressions: {x: {default_value: .inf}}");
    reject("invalid_track_reference", "ustx_version: '0.6'\ntracks: [{}]\nvoice_parts: [{track_no: 1}]");
    reject("negative_duration", "ustx_version: '0.6'\ntracks: [{}]\nvoice_parts: [{track_no: 0, notes: [{position: 0, duration: -1, tone: 60}]}]");
    reject("overflow_note_endpoint", "ustx_version: '0.6'\ntracks: [{}]\nvoice_parts: [{track_no: 0, position: 999999999, notes: [{position: 0, duration: 480, tone: 60}]}]");
    reject("mismatched_curve_points", "ustx_version: '0.6'\ntracks: [{}]\nvoice_parts: [{track_no: 0, curves: [{abbr: pitd, xs: [0,5], ys: [0]}]}]");
    reject("ambiguous_tempo_duplicate", "ustx_version: '0.6'\ntracks: [{}]\ntempos: [{position: 0, bpm: 120},{position: 0, bpm: 130}]");
    reject("byte_limit", std::string(16u * 1024u * 1024u + 1, ' '));
    reject("depth_limit", std::string(70, '[') + "0" + std::string(70, ']'));
    auto spline = actualSample;
    const auto shapePos = spline.find("shape: io");
    spline.replace(shapePos, std::string("shape: io").size(), "shape: sp");
    warnings.clear(); const auto sp = UstxImporter::parse(spline, error, warnings);
    check("spline_not_silently_replaced", sp && !sp->notes[0].hasPitchBend
        && warnings.joinIntoString("\n").contains("spline/unknown"));

    check("create_output_folder", folder.createDirectory().wasOk());
    const auto fixture = folder.getChildFile("upstream-note-wrapper.ustx");
    const auto bomSource = std::string("\xef\xbb\xbf") + actualSample;
    check("write_bom_fixture", fixture.replaceWithData(bomSource.data(), bomSource.size()));
    juce::MemoryBlock before, after; fixture.loadFileAsData(before);
    warnings.clear(); const auto disk = UstxImporter::read(fixture, error, warnings); fixture.loadFileAsData(after);
    check("read_bom_and_source_unchanged", disk && disk->sourceBom && before == after
        && std::string(disk->sourceText.toRawUTF8(), disk->sourceText.getNumBytesAsUTF8()) == bomSource);
    auto* report = new juce::DynamicObject;
    report->setProperty("ok", ok); report->setProperty("checks", checks); report->setProperty("failures", failures);
    report->setProperty("scope", "parser only; no PM conversion/render/build acceptance implied");
    report->setProperty("upstreamCommit", "33302b3d7cecf9ae830d4229639e06788f717b3c");
    report->setProperty("upstreamLicense", "MIT, Copyright (c) 2014 StAkira; notice in UstxImporterSmoke.h");
    report->setProperty("upstreamSource", "https://github.com/openutau/OpenUtau/blob/33302b3d7cecf9ae830d4229639e06788f717b3c/OpenUtau.Test/Core/USTx/UstxYamlTest.cs");
    const bool reportWritten = folder.getChildFile("verification.json").replaceWithText(juce::JSON::toString(juce::var(report), true));
    return ok && reportWritten;
}
}
