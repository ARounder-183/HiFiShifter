import { getQuickSearchInitialPosition } from "./quickSearchPosition.js";

function assertEqual(actual: unknown, expected: unknown, label: string): void {
    if (actual !== expected) {
        throw new Error(`${label}: expected ${String(expected)}, received ${String(actual)}`);
    }
}

const center = getQuickSearchInitialPosition({
    viewportWidth: 1000,
    viewportHeight: 800,
    pointer: null,
});

assertEqual(center.x, 340, "center x");
assertEqual(center.y, 200, "center y");

const pointer = getQuickSearchInitialPosition({
    viewportWidth: 1000,
    viewportHeight: 800,
    pointer: { x: 950, y: 760 },
});

assertEqual(pointer.x, 660, "pointer x clamped");
assertEqual(pointer.y, 400, "pointer y clamped");

console.log("quick search position checks passed");
