import { describe, expect, it } from "vitest";

import type { ParamViewSegment } from "./types";
import {
    buildParamBlockFrameCenters,
    resolveParamBlockFrameValue,
    resolveParamBlockFrameView,
    resolveParamBlockPanelValues,
} from "./paramBlockFrame";
import type { ParamBlock } from "./paramBlocks";

const authoritative: ParamViewSegment = {
    key: "v2|track|pitch|0|4|1",
    framePeriodMs: 5,
    startFrame: 0,
    stride: 1,
    referenceKind: "source_curve",
    orig: [60, 60, 60, 60],
    edit: [60, 60, 60, 60],
    edited: [],
};

describe("param block visual frame", () => {
    it("uses the interaction curve as the only source for lines and block sampling", () => {
        const interaction = { ...authoritative, edit: [64, 64, 64, 64] };
        const frameView = resolveParamBlockFrameView({
            interaction,
            live: {
                key: authoritative.key,
                edit: [62, 62, 62, 62],
            },
            paramView: authoritative,
        });

        expect(frameView).toBe(interaction);
    });

    it("derives an automatic block center from the current frame curve", () => {
        const interaction = { ...authoritative, edit: [64, 64, 64, 64] };
        const value = resolveParamBlockFrameValue({
            block: {
                startSec: 0,
                endSec: 0.02,
                averageValue: 58,
            },
            samples: {
                values: interaction.edit,
                startFrame: interaction.startFrame,
                stride: interaction.stride,
                framePeriodMs: interaction.framePeriodMs,
                isPitch: true,
                param: "pitch",
            },
        });

        expect(value).toBe(64);
    });

    it("uses the gesture frame center immediately while curve sampling catches up", () => {
        const value = resolveParamBlockFrameValue({
            block: {
                startSec: 0,
                endSec: 0.02,
                averageValue: 58,
                displayCenterValue: 48,
                frameCenterValue: 65,
            },
            samples: {
                values: authoritative.edit,
                startFrame: authoritative.startFrame,
                stride: authoritative.stride,
                framePeriodMs: authoritative.framePeriodMs,
                isPitch: true,
                param: "pitch",
            },
        });

        expect(value).toBe(65);
    });

    it("moves every selected block center by the same gesture delta", () => {
        const centers = buildParamBlockFrameCenters({
            blocks: [
                { id: "a", anchorValue: 60 },
                { id: "b", anchorValue: 64 },
            ],
            valueDelta: 2,
            param: "pitch",
            snapToSemitone: true,
        });

        expect([...centers]).toEqual([
            ["a", 62],
            ["b", 66],
        ]);
    });

    it("uses the derived fallback when no samples overlap", () => {
        const value = resolveParamBlockFrameValue({
            block: {
                startSec: 10,
                endSec: 11,
                averageValue: 58,
            },
            samples: {
                values: [64, 64, 64],
                startFrame: 0,
                stride: 1,
                framePeriodMs: 5,
                isPitch: true,
                param: "pitch",
            },
        });

        expect(value).toBe(58);
    });

    it("keeps an explicit drag target when a new sampled curve has a different cluster center", () => {
        expect(
            resolveParamBlockFrameValue({
                block: { startSec: 0, endSec: 0.02, averageValue: 58, logicalCenterValue: 62.5 },
                samples: {
                    values: [64, 64, 64, 64],
                    startFrame: 0,
                    stride: 1,
                    framePeriodMs: 5,
                    isPitch: true,
                    param: "pitch",
                },
            }),
        ).toBe(62.5);
    });

    it("honors the frozen center during a time resize or transition preview", () => {
        expect(
            resolveParamBlockFrameValue({
                block: { startSec: 0, endSec: 0.02, displayCenterValue: 61 },
                samples: {
                    values: [64, 64, 64, 64],
                    startFrame: 0,
                    stride: 1,
                    framePeriodMs: 5,
                    isPitch: true,
                    param: "pitch",
                },
            }),
        ).toBe(61);
    });

    it.each([
        ["volume", 1],
        ["dyn", 0.4],
        ["tension", 20],
        ["formant_shift_cents", -100],
        ["pitch_offset_cents", 50],
    ])("does not place a %s block at the pitch note's saved MIDI center", (param, value) => {
        expect(
            resolveParamBlockFrameValue({
                block: { startSec: 0, endSec: 0.02, averageValue: 64, logicalCenterValue: 64 },
                samples: {
                    values: [value, value, value, value],
                    startFrame: 0,
                    stride: 1,
                    framePeriodMs: 5,
                    isPitch: false,
                    param,
                },
            }),
        ).toBe(value);
    });

    it("keeps a persisted manual value for non-pitch user blocks", () => {
        const userBlock: ParamBlock = {
            id: "volume-block",
            rootTrackId: "track",
            kind: "user",
            startSec: 0,
            endSec: 0.02,
            members: [],
            averageValue: 0.3,
            manualValue: 0.8,
        };
        const volumeView = {
            ...authoritative,
            key: "v2|track|volume|0|4|1",
            edit: [0.3, 0.3, 0.3, 0.3],
        };
        const volume = resolveParamBlockPanelValues({
            blocks: [userBlock],
            param: "volume",
            paramView: volumeView,
            fallbackValue: 0,
        })[0];
        expect(volume.averageValue).toBe(0.8);
        expect(volume.manualValue).toBe(0.8);

        const autoBlock: ParamBlock = { ...userBlock, kind: "auto" };
        expect(
            resolveParamBlockPanelValues({
                blocks: [autoBlock],
                param: "volume",
                paramView: volumeView,
                fallbackValue: 0,
            })[0].averageValue,
        ).toBe(0.3);
    });

    it("derives each parameter from its own curve while preserving the pitch placement", () => {
        const block: ParamBlock = {
            id: "note",
            rootTrackId: "track",
            kind: "auto",
            startSec: 0,
            endSec: 0.02,
            members: [],
            averageValue: 64,
            manualValue: 64,
        };
        for (const [param, value] of [
            ["volume", 1],
            ["dyn", 0.4],
            ["tension", 20],
        ] as const) {
            const pv = {
                ...authoritative,
                key: `v2|track|${param}|0|4|1`,
                edit: [value, value, value, value],
            };
            const current = resolveParamBlockPanelValues({
                blocks: [block],
                param,
                paramView: pv,
                fallbackValue: 0,
            })[0];
            expect(current.averageValue).toBe(value);
            expect(current.manualValue).toBeNull();
        }
        const pitch = resolveParamBlockPanelValues({
            blocks: [block],
            param: "pitch",
            paramView: authoritative,
            fallbackValue: 60,
        })[0];
        expect(pitch.averageValue).toBe(64);
        expect(pitch.manualValue).toBe(64);
        expect(block.manualValue).toBe(64);
        expect(
            resolveParamBlockPanelValues({
                blocks: [block],
                param: "volume",
                paramView: null,
                fallbackValue: 1,
            })[0].averageValue,
        ).toBeNull();
        expect(
            resolveParamBlockPanelValues({
                blocks: [block],
                param: "volume",
                paramView: null,
                fallbackValue: 1,
                allowFallback: true,
            })[0].averageValue,
        ).toBe(1);
        expect(
            resolveParamBlockPanelValues({
                blocks: [block],
                param: "volume",
                paramView: null,
                fallbackValue: 1,
                previousValues: new Map([["note", { value: 0.75, startSec: 0, endSec: 0.02 }]]),
            })[0].averageValue,
        ).toBe(0.75);
        expect(
            resolveParamBlockPanelValues({
                blocks: [block],
                param: "volume",
                paramView: null,
                fallbackValue: 1,
                previousValues: new Map([["note", { value: 0.75, startSec: 0.1, endSec: 0.12 }]]),
            })[0].averageValue,
        ).toBeNull();
    });
});

it("does not project a previous parameter or root's preview into the new editor", () => {
    const stale = { ...authoritative, edit: [90, 90, 90, 90] };
    const volume = { ...authoritative, key: "v2|track|volume|0|4|1", edit: [1, 1, 1, 1] };
    expect(
        resolveParamBlockFrameView({
            interaction: stale,
            live: null,
            paramView: volume,
            param: "volume",
        }),
    ).toBe(volume);
    expect(
        resolveParamBlockFrameView({
            interaction: stale,
            live: null,
            paramView: null,
            param: "volume",
        }),
    ).toBeNull();
    const other = { ...authoritative, key: "v2|other|pitch|0|4|1" };
    expect(
        resolveParamBlockFrameView({
            interaction: stale,
            live: null,
            paramView: other,
            param: "pitch",
        }),
    ).toBe(other);
});

it("locked centerPolicy never falls back to curve sampling", () => {
    expect(
        resolveParamBlockFrameValue({
            block: {
                startSec: 0,
                endSec: 0.02,
                averageValue: 58,
                displayCenterValue: 61,
                centerPolicy: "locked",
            },
            samples: {
                values: [64, 64, 64, 64],
                startFrame: 0,
                stride: 1,
                framePeriodMs: 5,
                isPitch: true,
                param: "pitch",
            },
        }),
    ).toBe(61);
});
