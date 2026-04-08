import { markProjectDirty } from "./sessionDirtyState.js";

function assertEqual<T>(actual: T, expected: T): void {
    if (actual !== expected) {
        throw new Error(`Expected ${String(expected)}, received ${String(actual)}`);
    }
}

const project = { dirty: false };

markProjectDirty(project);

assertEqual(project.dirty, true);

console.log("session dirty state checks passed");
