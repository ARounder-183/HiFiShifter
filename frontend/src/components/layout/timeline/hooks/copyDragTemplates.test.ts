import { test } from "vitest";

import { buildCopyDragTemplates } from "./copyDragTemplates.js";

test("components/layout/timeline/hooks/copyDragTemplates.test.ts scripted checks", async () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        if (actual !== expected) {
            throw new Error(`${label}: expected ${String(expected)}, received ${String(actual)}`);
        }
    }

    const templates = await buildCopyDragTemplates({
        templateInputs: [
            {
                id: "clip-a",
                initial: { startSec: 1, trackId: "track-1" },
                now: {
                    name: "A",
                    lengthSec: 2,
                    sourcePath: "a.wav",
                    durationSec: 2,
                    gain: 1,
                    muted: false,
                    sourceStartSec: 0,
                    sourceEndSec: 2,
                    playbackRate: 1,
                    fadeInSec: 0,
                    fadeOutSec: 0,
                },
                targetTrackId: "track-2",
            },
        ],
        deltaSec: 3,
        linkedParamsResults: [{ ok: true, linkedParams: { pitch: [1, 2] } }],
    });

    assertEqual(templates[0]?.trackId, "track-2", "target track");
    assertEqual(templates[0]?.startSec, 4, "shifted start");
    assertEqual(
        Array.isArray(Object.values(templates[0]?.linkedParams ?? {})[0]),
        true,
        "linked params kept",
    );

});
