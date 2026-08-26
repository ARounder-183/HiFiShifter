import { test } from "vitest";

import { invoke } from "./invoke.ts";

test("services/invoke.bulkCommands.test.ts scripted checks", async () => {
    type TauriCall = {
        method: string;
        args: unknown;
    };

    function assertDeepEqual(actual: unknown, expected: unknown, label: string): void {
        const actualJson = JSON.stringify(actual);
        const expectedJson = JSON.stringify(expected);
        if (actualJson !== expectedJson) {
            throw new Error(`${label}: expected ${expectedJson}, received ${actualJson}`);
        }
    }

    async function expectTauriPayload(method: string, payload: unknown, expectedArgs: unknown) {
        const calls: TauriCall[] = [];
        Object.defineProperty(globalThis, "window", {
            configurable: true,
            writable: true,
            value: {
                __TAURI__: {
                    core: {
                        invoke: async <T>(cmd: string, args?: Record<string, unknown>) => {
                            calls.push({ method: cmd, args });
                            return { ok: true } as T;
                        },
                    },
                },
            } as Window & typeof globalThis,
        });

        await invoke(method, payload);

        assertDeepEqual(
            calls,
            [{ method, args: expectedArgs }],
            `${method} should forward named payload`,
        );
    }

    await expectTauriPayload(
        "create_clips_bulk",
        {
            templates: [{ trackId: "track-1", name: "clip", startSec: 0, lengthSec: 1 }],
            selectCreatedClips: true,
        },
        {
            payload: {
                templates: [{ trackId: "track-1", name: "clip", startSec: 0, lengthSec: 1 }],
                selectCreatedClips: true,
            },
        },
    );

    await expectTauriPayload(
        "duplicate_clips_bulk",
        {
            sourceClipIds: ["clip-1"],
            deltaSec: 2,
            trackMode: { kind: "same_track" },
            copyLinkedParams: true,
            selectCreatedClips: true,
        },
        {
            payload: {
                sourceClipIds: ["clip-1"],
                deltaSec: 2,
                trackMode: { kind: "same_track" },
                copyLinkedParams: true,
                selectCreatedClips: true,
            },
        },
    );

    await expectTauriPayload(
        "save_recording_settings",
        {
            sourceDevice: "default",
            sampleRate: 48_000,
            bitDepth: 24,
            channels: 2,
            inputGainDb: 0,
            monitorEnabled: false,
            monitorGainDb: 0,
            countdownSec: 0,
            autoNormalize: false,
            autoStopAtSelectionEnd: false,
            pathTemplate: "<ProjectFolder>/HiFiShifter Record/%Y-%m-%d-%H-%M-%S.wav",
        },
        {
            settings: {
                sourceDevice: "default",
                sampleRate: 48_000,
                bitDepth: 24,
                channels: 2,
                inputGainDb: 0,
                monitorEnabled: false,
                monitorGainDb: 0,
                countdownSec: 0,
                autoNormalize: false,
                autoStopAtSelectionEnd: false,
                pathTemplate: "<ProjectFolder>/HiFiShifter Record/%Y-%m-%d-%H-%M-%S.wav",
            },
        },
    );

    await expectTauriPayload("start_recording", 12.5, { startSec: 12.5 });

});
