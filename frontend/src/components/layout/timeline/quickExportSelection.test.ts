import { test } from "vitest";

import { buildQuickExportFileName, resolveQuickExportClipIds } from "./quickExportSelection.ts";

test("components/layout/timeline/quickExportSelection.test.ts scripted checks", async () => {
    function assertDeepEqual(actual: unknown, expected: unknown, label: string): void {
        const actualJson = JSON.stringify(actual);
        const expectedJson = JSON.stringify(expected);
        if (actualJson !== expectedJson) {
            throw new Error(`${label}: expected ${expectedJson}, received ${actualJson}`);
        }
    }

    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        if (actual !== expected) {
            throw new Error(`${label}: expected ${String(expected)}, received ${String(actual)}`);
        }
    }

    assertDeepEqual(
        resolveQuickExportClipIds({
            contextClipId: "clip-b",
            multiSelectedClipIds: ["clip-a", "clip-b", "clip-c"],
        }),
        ["clip-a", "clip-b", "clip-c"],
        "quick export uses active multi-selection when context clip is part of it",
    );

    assertDeepEqual(
        resolveQuickExportClipIds({
            contextClipId: "clip-z",
            multiSelectedClipIds: ["clip-a", "clip-b", "clip-c"],
        }),
        ["clip-z"],
        "quick export falls back to context clip when right-clicked clip is outside selection",
    );

    assertEqual(
        buildQuickExportFileName("Demo Project"),
        "Demo Project_quick_export.wav",
        "quick export file name uses project name",
    );

    assertEqual(
        buildQuickExportFileName(""),
        "quick_export.wav",
        "quick export file name falls back for empty project names",
    );

    console.log("quick export selection helpers passed");
});
