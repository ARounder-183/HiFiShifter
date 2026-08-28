import { test } from "vitest";

import { markProjectDirty } from "./sessionDirtyState.js";

test("features/session/sessionDirtyState.test.ts scripted checks", async () => {
    function assertEqual<T>(actual: T, expected: T): void {
        if (actual !== expected) {
            throw new Error(`Expected ${String(expected)}, received ${String(actual)}`);
        }
    }

    const project = { dirty: false };

    markProjectDirty(project);

    assertEqual(project.dirty, true);

});
