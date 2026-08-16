import { resolveClipDragCopyMode } from "./clipDragCopyMode.ts";

function assertEqual(actual: unknown, expected: unknown, label: string): void {
    if (actual !== expected) {
        throw new Error(`${label}: expected ${String(expected)}, received ${String(actual)}`);
    }
}

assertEqual(
    resolveClipDragCopyMode({
        existingCopyMode: false,
        ctrlKey: true,
        modifierActive: false,
    }),
    true,
    "ctrl starts copy drag",
);

assertEqual(
    resolveClipDragCopyMode({
        existingCopyMode: false,
        ctrlKey: false,
        modifierActive: true,
    }),
    true,
    "modifier binding can enable copy drag after pointer down",
);

assertEqual(
    resolveClipDragCopyMode({
        existingCopyMode: false,
        ctrlKey: false,
        modifierActive: false,
    }),
    false,
    "plain drag stays move drag",
);

console.log("clip drag copy mode checks passed");
