import { test } from "vitest";

import { resolveClipDragCopyMode } from "./clipDragCopyMode.ts";
import { IS_MAC } from "../../../../utils/platform.ts";

test("components/layout/timeline/hooks/clipDragCopyMode.test.ts scripted checks", async () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        if (actual !== expected) {
            throw new Error(`${label}: expected ${String(expected)}, received ${String(actual)}`);
        }
    }

    // 写死的 Ctrl 回退只在 Windows/Linux 生效：macOS 上 ctrl 字段映射为 ⌘，
    // 默认键位已覆盖 ⌘+拖拽，Ctrl 不再承担复制拖拽（见 clipDragCopyMode.ts
    // 的注释）。断言随平台变化，而不是写死 true。
    assertEqual(
        resolveClipDragCopyMode({
            existingCopyMode: false,
            ctrlKey: true,
            modifierActive: false,
        }),
        !IS_MAC,
        "ctrl starts copy drag (legacy fallback is Windows/Linux only)",
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
});
