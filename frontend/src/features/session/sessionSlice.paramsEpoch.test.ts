import { test } from "vitest";

import reducer, { bumpParamsEpoch } from "./sessionSlice.ts";

test("features/session/sessionSlice.paramsEpoch.test.ts scripted checks", async () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        const actualJson = JSON.stringify(actual);
        const expectedJson = JSON.stringify(expected);
        if (actualJson !== expectedJson) {
            throw new Error(`${label}: expected ${expectedJson}, received ${actualJson}`);
        }
    }

    const base = reducer(undefined, { type: "@@INIT" });
    const next = reducer(base, bumpParamsEpoch());

    assertEqual(
        next.paramsEpoch,
        Number(base.paramsEpoch) + 1,
        "bumpParamsEpoch increments the parameter refresh epoch",
    );
});
