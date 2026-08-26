import { test } from "vitest";

import { debounceMs } from "./useClipFormantEditor";

test("components/layout/timeline/clip/useClipFormantEditor.test.ts scripted checks", async () => {
    if (debounceMs() !== 180) {
        throw new Error(`expected debounceMs() to equal 180, got ${debounceMs()}`);
    }
});
