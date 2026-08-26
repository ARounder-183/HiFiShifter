import { test } from "vitest";

import { getClipFormantButtonStyle } from "./clipFormantButtonStyle";

test("components/layout/timeline/clip/clipFormantButtonStyle.test.ts scripted checks", async () => {
    function assert(condition: unknown, message: string): void {
        if (!condition) {
            throw new Error(message);
        }
    }

    const base = {
        baseBackgroundColor: "rgba(1, 2, 3, 0.4)",
        baseBorderColor: "rgba(4, 5, 6, 0.5)",
        baseTextColor: "rgba(7, 8, 9, 0.6)",
    };

    const idleStyle = getClipFormantButtonStyle({
        ...base,
        enabled: false,
        status: "ready",
    });

    assert(
        idleStyle.backgroundColor === base.baseBackgroundColor,
        "idle bg should match mute badge",
    );
    assert(idleStyle.borderColor === base.baseBorderColor, "idle border should match mute badge");
    assert(idleStyle.color === base.baseTextColor, "idle text should match mute badge");

    const enabledStyle = getClipFormantButtonStyle({
        ...base,
        enabled: true,
        status: "ready",
    });

    assert(
        enabledStyle.backgroundColor !== base.baseBackgroundColor,
        "enabled bg should tint the base badge",
    );

    const failedStyle = getClipFormantButtonStyle({
        ...base,
        enabled: true,
        status: "failed",
    });

    assert(
        failedStyle.backgroundColor.includes("var(--qt-danger-bg)"),
        "failed state should use danger palette",
    );

});
