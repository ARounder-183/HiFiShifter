import { test } from "vitest";

import { classifyClipEditFailure } from "./clipEditFailure.ts";

/**
 * **不变量：只有需要用户处置的编辑失败才提示"音频块编辑被拒绝"。**
 *
 * 【为什么这是回归测试】`setClipStateRemote` / `setClipsStateBulkRemote` 的 `.rejected`
 * 此前一律把状态置成 `"Clip edit rejected"`，而**任何**失败模式都走同一条路 —— 包括
 * "宿主还在追上几何/媒体"这种后端会自己重试并收敛的瞬时失败。用户因此每做一次编辑都
 * 被报一次错，即使编辑其实生效了（用户报障："执行任何编辑操作时，总是会闪烁地提示…"
 * 与"一直触发音频块编辑被拒绝"）。
 */
test("features/session/clipEditFailure.test.ts scripted checks", () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        if (actual !== expected) {
            throw new Error(`${label}: expected ${String(expected)}, received ${String(actual)}`);
        }
    }

    // 后端在宿主回调边界产出的**瞬时**文案：插件会自行重试并收敛。
    for (const message of [
        "clip was edited in REAPER but its ARA geometry has not caught up; host Undo remains available",
        "item was split in REAPER but both clips have not caught up; host Undo remains available",
        "item was created in REAPER but its ARA audio is not ready; use host Undo if canceling",
        "Conflict: host changed; local curves preserved, host is not ready",
        "host clip changed before GUI commit; refresh required",
        "native host edit request budget exceeded",
        "host editor lease revoked",
    ]) {
        assertEqual(
            classifyClipEditFailure(message),
            "transient",
            `瞬时失败不得提示用户：${message}`,
        );
    }

    // 需要用户处置的失败仍然必须可见。
    for (const message of [
        "host clip property is not yet writable: color",
        "nonlinear host stretch-marker editing is not supported",
        "unknown editor clip",
        "clip has no unique directly bound REAPER take",
    ]) {
        assertEqual(
            classifyClipEditFailure(message),
            "permanent",
            `需要处置的失败必须提示：${message}`,
        );
    }

    // 未知/空文案默认按"永久"：宁可多报一次，也不能把真正的失败静默掉。
    assertEqual(classifyClipEditFailure("something new went wrong"), "permanent", "未知文案");
    assertEqual(classifyClipEditFailure(""), "permanent", "空文案");
    assertEqual(classifyClipEditFailure(undefined), "permanent", "缺失文案");
});
