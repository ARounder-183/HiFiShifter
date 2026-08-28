import { test } from "vitest";

import { resolveTimelineClipHeaderVisibility } from "./timelineClipHeaderVisibility.js";

test("components/layout/timeline/runtime/timelineClipHeaderVisibility.test.ts scripted checks", async () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        const actualJson = JSON.stringify(actual);
        const expectedJson = JSON.stringify(expected);
        if (actualJson !== expectedJson) {
            throw new Error(`${label}: expected ${expectedJson}, received ${actualJson}`);
        }
    }

    assertEqual(
        resolveTimelineClipHeaderVisibility(24),
        {
            showAny: false,
            showChain: false,
            showMute: false,
            showFormant: false,
            showGainKnob: false,
            showGainLabel: false,
            showPlaybackRate: false,
            showName: false,
        },
        "very narrow clips hide header contents",
    );

    assertEqual(
        resolveTimelineClipHeaderVisibility(40),
        {
            showAny: true,
            showChain: true,
            showMute: false,
            showFormant: false,
            showGainKnob: true,
            showGainLabel: false,
            showPlaybackRate: false,
            showName: false,
        },
        "narrow clips prioritize gain knob before mute",
    );

    assertEqual(
        resolveTimelineClipHeaderVisibility(56),
        {
            showAny: true,
            showChain: true,
            showMute: true,
            showFormant: false,
            showGainKnob: true,
            showGainLabel: false,
            showPlaybackRate: false,
            showName: false,
        },
        "medium clips keep mute and gain knob visible",
    );

    assertEqual(
        resolveTimelineClipHeaderVisibility(120),
        {
            showAny: true,
            showChain: true,
            showMute: true,
            showFormant: true,
            showGainKnob: true,
            showGainLabel: true,
            showPlaybackRate: true,
            showName: false,
        },
        "playback rate appears before name when width is limited",
    );

    assertEqual(
        resolveTimelineClipHeaderVisibility(160),
        {
            showAny: true,
            showChain: true,
            showMute: true,
            showFormant: true,
            showGainKnob: true,
            showGainLabel: true,
            showPlaybackRate: true,
            showName: true,
        },
        "wide clips keep full header contents visible",
    );

    assertEqual(
        resolveTimelineClipHeaderVisibility(160, true),
        {
            showAny: true,
            showChain: true,
            showMute: true,
            showFormant: false,
            showGainKnob: false,
            showGainLabel: true,
            showPlaybackRate: true,
            showName: true,
        },
        "pitch-adjustment clips hide formant badge and gain knob",
    );
});
